"""The host's single pytest slot: agentctl's ``pytest`` pool.

pytest is the heaviest thing this checkout runs, and several agent sessions
share one workstation. A run started from a session subagent sits outside every
load control the runtime applies to its own jobs, so concurrent runs contend
for the same cores and disk until a long job passes its timeout.

Every managed pytest run therefore runs inside the host's ``pytest`` pool (one
task at a time). A process already inside that pool holds the slot: the
runtime places the job in ``agentctl-pytest.slice`` (``sinnixd-pueue-pytest.slice``
on older hosts) and exports the pool name. A job id alone is not ownership:
lane jobs and every other caller submit the declared ``pytest_focused``
operation through ``agentctl job start`` and wait for it.
``POLYLOGUE_PYTEST_SLOT=held`` is the explicit escape for a hermetic test of
this mechanism.

The managed pytest environment travels in a launch file the queued operation
consumes, because the queue persists the submitting client's environment and
the declared operation resolves its own.

Every run started here also keeps its temporary trees inside the checkout and
disposes of them on success, while failures and interruptions retain their
diagnostic trees and telemetry; see :func:`basetemp_root`.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final

from devtools.agent_env import PYTEST_POOL, PYTEST_POOLS, inside_pytest_pool
from devtools.cloud_sentinels import cloud_sentinel_declined
from devtools.pytest_memory import CUSTODY_ENV, ProcessGroupMemorySampler
from devtools.pytest_memory_admission import (
    EX_TEMPFAIL,
    RESOURCE_NOT_READY,
    admission_ledger,
    admission_not_ready,
    admit_width,
)
from devtools.worker_memory import ChargeProfile, charge_profile_for, corroborate_profile, resize_worker_argument

__all__ = [
    "BASETEMP_ROOT_ENV",
    "INHERITED_ENVIRONMENT_KEYS",
    "REAPED_SIGNALS",
    "PYTEST_OPERATION",
    "PYTEST_POOL",
    "PytestSlotUnavailableError",
    "SlotOutcome",
    "basetemp_root",
    "client_environment",
    "contained_pytest_run",
    "guard_temp_trees",
    "holds_pytest_slot",
    "main",
    "remove_temp_tree",
    "run_pytest_isolated",
    "run_pytest",
    "sweep_stale_temp_trees",
]

#: The runtime's command line.
AGENTCTL: Final = "agentctl"
#: The declared operation in the pytest pool that runs one launch file.
PYTEST_OPERATION: Final = "pytest_focused"
#: The runtime's executor, as it appears in a job's process ancestry.
QUEUE_RUNNERS: Final = ("agentctl-run", "sinnixd-queue-run")
#: Explicit escape, for the hermetic test of this mechanism.
SLOT_ESCAPE_ENV: Final = "POLYLOGUE_PYTEST_SLOT"
SLOT_HELD: Final = "held"
#: How often the waiter asks the runtime whether its job has ended.
POLL_INTERVAL_S: Final = 2.0

#: Signals that end this process while it waits on a job it owns. The job
#: outlives the waiter, and the pool's parallelism is one, so an unreaped job
#: starves every other checkout on the host until someone notices.
REAPED_SIGNALS: Final[tuple[signal.Signals, ...]] = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

#: The whole budget a signalled run has before the unit is SIGKILLed:
#: ``systemctl --user show -p DefaultTimeoutStopUSec`` on this host. Nothing
#: here enforces it -- it is the ceiling everything below is sized against,
#: and if the runtime's default moves, this line must move with it.
UNIT_STOP_BUDGET_S: Final = 15.0
#: How long the child process group gets to end on SIGTERM, and then on
#: SIGKILL, before the signal is re-raised into the caller's own handler.
#:
#: These are deliberately small. The reap runs FIRST (see :func:`_on_exit`),
#: so every second it spends is a second the receipt-writing handler above it
#: does not have. The shipped form was 5 + 5, and on 2026-09-20 a cancelled
#: corpus run spent 10.01s of a 15s budget here (Stopping 03:05:26.885 ->
#: Stopped 03:05:36.898) and the interrupted receipt never landed: what was
#: left had to cover finish_interrupted_steps, append_verify_history,
#: append_verification_evidence, prune_successful_verify_runs and
#: write_failure_seed on a host whose io_full_avg10 had just been flagged at
#: 26.1. Reaping a child is cheap to retry from outside; a receipt that was
#: never written is not recoverable at all.
#:
#: Both legs are spent in full, not as a worst case. ``stop()`` runs inside a
#: signal handler nested in the process's own blocking ``Popen.wait()``, which
#: holds ``_waitpid_lock``; the nested ``wait(timeout=...)`` can never acquire
#: it, so each leg spins to its own timeout even when the child died at once.
#: Measured here: 4.015s for 2 + 2, against the incident's 10.01s for 5 + 5.
STOP_TERM_GRACE_S: Final = 2.0
STOP_KILL_GRACE_S: Final = 2.0
#: What the reap may spend of the unit's stop budget in the worst case. The
#: majority of :data:`UNIT_STOP_BUDGET_S` must remain for the outer handler.
STOP_ESCALATION_BUDGET_S: Final = STOP_TERM_GRACE_S + STOP_KILL_GRACE_S

#: The only keys the ``agentctl`` client inherits: what the runtime needs to
#: reach the queue and enter the project environment. Everything pytest needs
#: travels in the launch file, because the queue persists the client's
#: environment into shared state.
INHERITED_ENVIRONMENT_KEYS: Final[tuple[str, ...]] = (
    "HOME",
    "USER",
    "PATH",
    "LANG",
    "TERM",
    "SSH_AUTH_SOCK",
    "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
)

LAUNCH_DIR: Final = Path(".cache/verify")
ISOLATED_ARCHIVE_PARENT: Final = Path("/realm/tmp/work/polylogue-pytest-archives")

REFUSAL = (
    "pytest could not acquire the host's pytest slot: {reason}. "
    "Start the queue with `systemctl --user start pueued`, then rerun. "
    "Set POLYLOGUE_PYTEST_SLOT=held only when the caller already holds the slot."
)


class PytestSlotUnavailableError(RuntimeError):
    """The pytest slot could not be acquired, and the run must not proceed."""

    runtime_evidence: dict[str, Any] | None = None


@dataclass(frozen=True)
class SlotOutcome:
    returncode: int
    #: What the receipt records: ``agentctl job 12`` or ``held``.
    slot: str
    #: Where the queued run's output landed, or None when it streamed.
    log_path: Path | None = None
    receipt: dict[str, Any] | None = None
    #: How the queued job's unit was ended, when something outside the run
    #: ended it: ``{"killer": "oom-kill", "unit": ...}`` from AgentCTL's
    #: outcome record.
    termination: dict[str, Any] | None = None


#: The diagnosis of a queued run systemd-oomd killed.
OOM_KILLED_DIAGNOSIS = "oom_killed"


def _job_termination(reference: str | None) -> dict[str, Any] | None:
    """The killer AgentCTL recorded for a finished job's unit, if any.

    An oomd kill takes the whole unit, including the in-unit writer of the
    slot receipt, so this record is the only evidence of the cause.
    """
    from devtools.verify_runs import read_agentctl_outcome

    outcome = read_agentctl_outcome(reference) if reference else None
    if outcome is None:
        return None
    killer = outcome.get("systemd_result")
    if not isinstance(killer, str) or not killer:
        return None
    unit = outcome.get("unit")
    return {"killer": killer, "unit": unit if isinstance(unit, str) else None}


def termination_metadata(outcome: SlotOutcome) -> dict[str, Any]:
    """Step fields recording execution authority and external termination.

    A systemd-oomd kill is the typed ``oom_killed`` diagnosis, not the missing
    report or bare exit 137 it otherwise leaves.
    """
    metadata: dict[str, Any] = {}
    if isinstance(outcome.receipt, Mapping) and isinstance(outcome.receipt.get("execution_source"), Mapping):
        metadata["execution_source"] = outcome.receipt["execution_source"]
        if outcome.receipt["execution_source"].get("status") != "stable":
            metadata.update({"diagnosis": "execution_source_unavailable", "worktree_provenance_unknown": True})
    termination = outcome.termination
    if not termination:
        return metadata
    killer = termination.get("killer")
    metadata.update({"termination_killer": killer, "termination_unit": termination.get("unit")})
    if killer == "oom-kill":
        metadata.update({"diagnosis": OOM_KILLED_DIAGNOSIS, "termination_reason": OOM_KILLED_DIAGNOSIS})
    return metadata


#: Names the directory managed pytest runs put their temporary trees under.
BASETEMP_ROOT_ENV: Final = "POLYLOGUE_PYTEST_BASETEMP_ROOT"


def basetemp_root(env: Mapping[str, str], *, root: Path) -> Path:
    """The directory a managed pytest run puts its temporary trees under.

    pytest's default basetemp follows TMPDIR, which ``nix develop`` points at a
    per-shell directory on the host's small ``/tmp`` tmpfs; a corpus run fills
    the mount and dies on exhausted space or file descriptors. The checkout's
    own scratch directory sits on the same disposable filesystem as the rest of
    the verification artifacts and is sized for it.

    ``POLYLOGUE_PYTEST_BASETEMP_ROOT`` names a different root, for a sandbox
    with no checkout-local scratch. Its cloud sentinel value leaks into
    workstation agent sessions through ``.claude/settings.json`` and is
    declined there, since honouring it is exactly the tmpfs failure above.
    """
    configured = env.get(BASETEMP_ROOT_ENV)
    if configured and not cloud_sentinel_declined(BASETEMP_ROOT_ENV, configured):
        return Path(configured)
    return root / LAUNCH_DIR


def remove_temp_tree(path: Path) -> None:
    """Delete a pytest temporary tree, including the read-only trees tests seal.

    Sealed archive generations are written without write permission, so a plain
    rmtree cannot unlink them and silently leaves the tree behind. Unlinking
    needs a writable parent directory, not writable files; changing a regular
    file's mode can also mutate a retained fixture when the tree contains a
    hard link to it.
    """
    for parent, directories, _files in os.walk(path, topdown=False):
        for name in directories:
            directory = os.path.join(parent, name)
            with contextlib.suppress(OSError):
                mode = os.lstat(directory).st_mode
                if stat.S_ISDIR(mode):
                    os.chmod(directory, mode | stat.S_IWUSR)
        with contextlib.suppress(OSError):
            mode = os.lstat(parent).st_mode
            if stat.S_ISDIR(mode):
                os.chmod(parent, mode | stat.S_IWUSR)
    shutil.rmtree(path, ignore_errors=True)


#: ``tmp-<pid>-<nanoseconds hex>`` -- the name every managed run gives its
#: basetemp, and the only thing that records which process owned the tree.
_TEMP_TREE_NAME = re.compile(r"^tmp-(?P<pid>\d+)-[0-9a-f]+(?:\.tmpdir)?$")


def _process_is_live(pid: int, *, proc: Path = Path("/proc")) -> bool:
    """Whether a pid is still running, read from ``/proc`` when it exists."""
    if pid <= 0:
        return False
    if proc.is_dir():
        return (proc / str(pid)).exists()
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def sweep_stale_temp_trees(root_dir: Path, *, proc: Path = Path("/proc")) -> tuple[Path, ...]:
    """Remove ``tmp-<pid>-*`` trees under ``root_dir`` whose owning pid is gone.

    A run that is killed outright (SIGKILL, preemption, a dead container) runs
    no handler at all, so its basetemp survives every in-process disposal
    route. Evidence for the sweep: 514 such trees across this host's worktrees,
    56.7 GiB, all owned by pids that had not existed for weeks. A live pid's
    tree is never touched -- concurrent runs share this root.
    """
    removed: list[Path] = []
    own = os.getpid()
    try:
        entries = sorted(root_dir.iterdir())
    except OSError:
        return ()
    for entry in entries:
        match = _TEMP_TREE_NAME.match(entry.name)
        if match is None or not entry.is_dir():
            continue
        pid = int(match.group("pid"))
        if pid == own or _process_is_live(pid, proc=proc):
            continue
        remove_temp_tree(entry)
        removed.append(entry)
    return tuple(removed)


class _TempTreeGuard:
    """Removes its trees on any exit path until the owner cancels it.

    The ordinary route disposes of a tree in a ``finally``; that covers a
    return and an exception but not a terminating signal, which is exactly how
    a cancelled or preempted run ends. The guard adds ``atexit`` plus the
    reaped signals, and re-raises the signal with its previous disposition so
    the process still dies the way its caller asked it to.
    """

    def __init__(self, paths: Sequence[Path]) -> None:
        self._paths = tuple(paths)
        self._cancelled = False
        self._previous: dict[signal.Signals, Any] = {}
        atexit.register(self._dispose)
        for number in REAPED_SIGNALS:
            try:
                self._previous[number] = signal.signal(number, self._on_signal)
            except (OSError, ValueError):  # not the main thread, or not supported here
                continue

    def _dispose(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        for path in self._paths:
            remove_temp_tree(path)

    def _on_signal(self, signum: int, frame: Any) -> None:
        # Read the previous disposition before restoring: an outer handler may
        # have re-installed this one after cancelling the guard, and re-raising
        # into a handler that is still installed loops forever.
        previous = self._previous.get(signal.Signals(signum), signal.SIG_DFL)
        self._dispose()
        self._restore()
        if callable(previous):
            previous(signum, frame)
            return
        with contextlib.suppress(OSError, ValueError):
            signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)

    def _restore(self) -> None:
        for number, handler in self._previous.items():
            with contextlib.suppress(OSError, ValueError):
                signal.signal(number, handler)
        self._previous.clear()

    def cancel(self) -> None:
        """Stop guarding: the owner has decided this tree's disposition."""
        self._cancelled = True
        self._restore()
        with contextlib.suppress(Exception):
            atexit.unregister(self._dispose)


def guard_temp_trees(*paths: Path) -> _TempTreeGuard:
    """Guard ``paths`` against a run that never reaches its own cleanup."""
    return _TempTreeGuard(paths)


def _declared_basetemp(command: Sequence[str]) -> str | None:
    for index, argument in enumerate(command):
        if argument == "--basetemp" and index + 1 < len(command):
            return command[index + 1]
        if argument.startswith("--basetemp="):
            return argument.split("=", 1)[1]
    return None


def contained_pytest_run(
    command: Sequence[str], *, env: Mapping[str, str], root: Path
) -> tuple[list[str], dict[str, str], Path]:
    """``command`` and ``env`` with every temporary tree inside the checkout.

    Both halves are needed: ``--basetemp`` covers the fixtures pytest hands
    out, and TMPDIR covers what the code under test asks the standard library
    for. Returns the scratch directory the caller owns; the basetemp is its
    sibling without the ``.tmpdir`` suffix.
    """
    argv = list(command)
    declared = _declared_basetemp(argv)
    if declared is None:
        basetemp = basetemp_root(env, root=root) / f"tmp-{os.getpid()}-{time.time_ns():x}"
        argv += ["--basetemp", str(basetemp)]
    else:
        basetemp = Path(declared)
    if not basetemp.is_absolute():
        basetemp = root / basetemp
    # pytest empties its own basetemp as it starts, so TMPDIR is its sibling.
    scratch = basetemp.parent / f"{basetemp.name}.tmpdir"
    scratch.mkdir(parents=True, exist_ok=True)
    contained = dict(env)
    contained.update({"TMPDIR": str(scratch), "TMP": str(scratch), "TEMP": str(scratch)})
    # The temporary trees live inside the checkout. Repository discovery walks
    # upward, so a directory a test builds to be outside any repository would
    # otherwise be inside this one; the ceiling stops discovery below the
    # checkout while the trees themselves stay searchable.
    ceiling = str(basetemp.parent.resolve())
    inherited = env.get("GIT_CEILING_DIRECTORIES", "")
    contained["GIT_CEILING_DIRECTORIES"] = f"{ceiling}:{inherited}" if inherited else ceiling
    return argv, contained, scratch


def _process_ancestry(pid: int, *, proc: Path) -> Iterator[int]:
    """This process and each of its parents, ending at pid 1 or an unreadable one."""
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        yield pid
        try:
            fields = (proc / str(pid) / "stat").read_text(encoding="utf-8").rpartition(")")[2].split()
        except OSError:
            return
        try:
            pid = int(fields[1])
        except (IndexError, ValueError):
            return


def _launch_document_of(pid: int, *, proc: Path) -> Path | None:
    """The launch document a queue-runner process was started with, if this is one."""
    try:
        argv = (proc / str(pid) / "cmdline").read_bytes().decode("utf-8", "replace").split("\0")
    except OSError:
        return None
    words = [word for word in argv if word]
    if not any(Path(word).name.removeprefix(".").removesuffix("-wrapped") in QUEUE_RUNNERS for word in words):
        return None
    return Path(words[-1]) if words and words[-1].endswith(".json") else None


def declared_pool_of_enclosing_job(env: Mapping[str, str], *, proc: Path = Path("/proc")) -> str | None:
    """The pool declared by the queue task this process runs inside, if any.

    The queue runner exports nothing that names the job's pool, and the pool's
    systemd slice is not visible from every host configuration, so neither the
    environment nor the cgroup classifies the process reliably. The launch
    document does: the runner is an ancestor of this process and carries the
    document's path as its final argument, and the document declares ``pool``.
    """
    del env
    for pid in _process_ancestry(os.getpid(), proc=proc):
        document_path = _launch_document_of(pid, proc=proc)
        if document_path is None:
            continue
        try:
            document = json.loads(document_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pool = document.get("pool") if isinstance(document, Mapping) else None
        if isinstance(pool, str):
            return pool
    return None


def holds_pytest_slot(
    env: Mapping[str, str],
    *,
    cgroup_reader: Callable[[], str] | None = None,
    proc: Path = Path("/proc"),
) -> bool:
    """Whether this process is already inside the host's pytest slot.

    Ownership is the pytest pool: the cgroup the runtime placed the job in, the
    pool name it exported, or the pool the enclosing job's launch document
    declares. A job id alone never is. Enqueueing from inside the pool waits on
    a single-slot group only this job can drain, which is a deadlock.
    """
    if env.get(SLOT_ESCAPE_ENV) == SLOT_HELD or inside_pytest_pool(env, cgroup_reader=cgroup_reader):
        return True
    return declared_pool_of_enclosing_job(env, proc=proc) in PYTEST_POOLS


def client_environment(env: Mapping[str, str]) -> dict[str, str]:
    """The reduced environment the ``agentctl`` client runs with."""
    return {key: env[key] for key in INHERITED_ENVIRONMENT_KEYS if env.get(key)}


def _agentctl(arguments: Sequence[str], *, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    executable = shutil.which(AGENTCTL, path=env.get("PATH") or os.defpath)
    if executable is None:
        raise PytestSlotUnavailableError(REFUSAL.format(reason=f"`{AGENTCTL}` is not on PATH"))
    try:
        return subprocess.run(
            [executable, *arguments],
            env=dict(env),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise PytestSlotUnavailableError(
            REFUSAL.format(reason=f"`{AGENTCTL} {' '.join(arguments)}` could not start: {exc}")
        ) from exc


def _document(completed: subprocess.CompletedProcess[str], *, verb: str) -> dict[str, Any]:
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PytestSlotUnavailableError(REFUSAL.format(reason=f"`{AGENTCTL} {verb}` failed: {detail}"))
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PytestSlotUnavailableError(
            REFUSAL.format(reason=f"`{AGENTCTL} {verb}` printed no document: {exc}")
        ) from exc
    if not isinstance(document, dict):
        raise PytestSlotUnavailableError(REFUSAL.format(reason=f"`{AGENTCTL} {verb}` printed no document"))
    return document


def _cancel_job(job_id: int, *, env: Mapping[str, str], reference: str | None = None) -> bool:
    """Cancel through the owner that also stops the job's transient unit."""
    completed = _agentctl(
        ["--json", "job", "cancel", str(job_id), *(["--reference", reference] if reference else [])], env=env
    )
    document = _document(completed, verb="job cancel")
    # A zero process exit only means the cancel request was handled. In
    # particular, state=unresolved means agentctl could not establish that the
    # queue or its runner stopped consuming the launch file.
    return document.get("state") in {"removed", "stopped", "terminal"}


#: Longest pause between reads while the job cannot be observed.
OBSERVATION_BACKOFF_MAX_S: Final = 60.0
#: How often an unobservable job is reported as stalled.
STALL_REPORT_INTERVAL_S: Final = 300.0


def _wait_for(job_id: int, *, reference: str | None, env: Mapping[str, str]) -> dict[str, Any]:
    """The job's terminal view. The queue wait has no deadline; the job itself has one.

    Only a terminal view ends the wait. A read that fails -- the runtime being
    re-activated, a secret briefly unreadable -- is transient: the job is
    still queued or running, and giving up would abandon a slot it may have
    waited an hour for. Reads back off and the stall is reported instead. The
    launch reference addresses the job even after pueue drops its entry.
    """
    command = ["--json", "job", "get", str(job_id), *(["--reference", reference] if reference else [])]
    failures = 0
    stalled_since: float | None = None
    last_report = 0.0
    while True:
        problem: str | None = None
        view: dict[str, Any] = {}
        try:
            view = _document(_agentctl(command, env=env), verb="job get")
        except PytestSlotUnavailableError as exc:
            problem = str(exc)
        if problem is None and (
            view.get("job_id") != job_id
            or not isinstance(view.get("terminal"), bool)
            or not isinstance(view.get("phase"), str)
        ):
            problem = f"`{AGENTCTL} job get` returned an invalid view for job {job_id}"
        if problem is None:
            if stalled_since is not None:
                sys.stderr.write(f"  {AGENTCTL} job {job_id} observable again\n")
                sys.stderr.flush()
            failures, stalled_since = 0, None
            if view.get("terminal"):
                return view
            time.sleep(POLL_INTERVAL_S)
            continue
        failures += 1
        now = time.monotonic()
        if stalled_since is None:
            stalled_since = now
        if now - last_report >= STALL_REPORT_INTERVAL_S or failures == 1:
            last_report = now
            sys.stderr.write(
                f"  cannot observe {AGENTCTL} job {job_id} for {now - stalled_since:.0f}s "
                f"(still waiting; it is not cancelled): {problem[:300]}\n"
            )
            sys.stderr.flush()
        time.sleep(min(POLL_INTERVAL_S * 2 ** min(failures, 16), OBSERVATION_BACKOFF_MAX_S))


#: Terminal phases ``agentctl job get`` reports for a job that never ran its
#: command to an end of its own, with what each one means to the caller.
_DID_NOT_RUN: Final[dict[str, str]] = {
    "cancelled": "the job was cancelled",
    "vanished": "the job vanished before it finished",
    "slot_occupied": "the pytest pool was occupied and the runtime did not retry",
    "refused": "the runtime refused the job",
    "dependency-failed": "a job it depended on failed",
    "launch-failed": "the job could not be launched",
}


def _job_exit_status(view: Mapping[str, Any], *, receipt: Mapping[str, Any] | None) -> int:
    phase = view.get("phase")
    exit_code = view.get("exit_code")
    status = exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None
    if phase == "succeeded":
        return 0
    if phase == "failed":
        return status or 1
    if phase == "timeout" or (receipt is not None and receipt.get("status") == "timed_out"):
        return 124
    job = f"{AGENTCTL} job {view.get('job_id')}"
    meaning = _DID_NOT_RUN.get(str(phase))
    detail = f"{job} ended {phase!r} (exit {status!r})"
    if meaning is not None:
        detail = f"{meaning}: {detail}"
    raise PytestSlotUnavailableError(REFUSAL.format(reason=detail))


def _reap_job(
    job_id: int, *, env: Mapping[str, str], launch_path: Path | None = None, reference: str | None = None
) -> bool:
    """End a job this process owns and stop its transient unit.

    Best effort by construction: the reason we are here is that the waiter is
    being killed, so a failing reap must not replace the original cause of
    death with its own error.

    The launch file belongs to whoever ends the run, so deleting it is part of
    a cancellation that succeeded: a job still on the queue reads it when it
    starts.
    """
    cancelled = False
    with contextlib.suppress(PytestSlotUnavailableError):
        cancelled = _cancel_job(job_id, env=env, reference=reference)
    if cancelled and launch_path is not None:
        with contextlib.suppress(OSError):
            launch_path.unlink(missing_ok=True)
    return cancelled


@contextlib.contextmanager
def _on_exit(*actions: Callable[[], None], on_signal: Callable[[int], None] | None = None) -> Iterator[None]:
    """Run ``actions`` if this process is signalled or unwound inside the block.

    A signal runs the actions, restores the previous handler and re-raises the
    signal, so an outer handler (or the default action) decides what the
    signal means; the actions are what must not be skipped on the way there.

    They also run BEFORE that outer handler, which makes their duration a tax
    on it. The caller's handler is what writes the run's terminal receipt, and
    the whole sequence shares one unit stop budget, so an action here must be
    bounded well inside :data:`UNIT_STOP_BUDGET_S` -- see
    :data:`STOP_ESCALATION_BUDGET_S`. An action that can block is a receipt
    that does not get written.

    ``on_signal`` runs first and is told which signal arrived, so a caller can
    preserve the run's own evidence before the disposal actions remove the
    scratch it was sampled from. It runs on the signal path only: an ordinary
    unwind reaches the caller's own ``finally``.
    """

    def run_actions() -> None:
        for action in actions:
            with contextlib.suppress(Exception):
                action()

    def handle(signal_number: int, frame: object) -> None:
        if on_signal is not None:
            with contextlib.suppress(Exception):
                on_signal(signal_number)
        run_actions()
        signal.signal(signal_number, previous.get(signal.Signals(signal_number), signal.SIG_DFL))
        os.kill(os.getpid(), signal_number)

    previous: dict[signal.Signals, Any] = {}
    for number in REAPED_SIGNALS:
        with contextlib.suppress(ValueError, OSError):
            previous[number] = signal.signal(number, handle)
    try:
        yield
    except BaseException:
        run_actions()
        raise
    finally:
        for number, handler in previous.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(number, handler)


def _write_launch(
    path: Path,
    *,
    argv: Sequence[str],
    cwd: str,
    env: Mapping[str, str],
    log_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    worker_environment = dict(env)
    # The queue snapshot must not serialize an operator archive path. The
    # receiving worker installs a private scratch root before running pytest.
    worker_environment.pop("POLYLOGUE_ARCHIVE_ROOT", None)
    document = {
        "kind": "polylogue.pytest-slot-launch",
        "argv": list(argv),
        "working_directory": cwd,
        "environment": worker_environment,
        "log_path": str(log_path),
    }
    # The launch file carries the resolved environment; keep it off other users.
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump(document, stream)


def _submit(
    command: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    root: Path,
    on_exit: Callable[[], None],
    resource_state: dict[str, bool] | None = None,
    preserve_guard: Callable[[], None] | None = None,
) -> SlotOutcome:
    """Run ``command`` as the declared pytest-pool operation and wait for it."""
    identity = f"{os.getpid()}-{time.time_ns():x}"
    launch_path = root / LAUNCH_DIR / f"pytest-slot-{identity}.json"
    log_path = root / LAUNCH_DIR / f"pytest-slot-{identity}.log"
    client = client_environment(env)
    # The paths are per client pid, so a previous run of this pid may have left
    # a result document; reading that one would report someone else's run.
    _slot_result_path(log_path).unlink(missing_ok=True)
    _write_launch(launch_path, argv=command, cwd=cwd, env=env, log_path=log_path)
    launched: dict[str, Any] = {}

    def reap_owned_job() -> None:
        # Only on this waiter's own death (a signal or interpreter exit): the
        # job is then cancelled, by reference so it is found even if the queue
        # dropped its entry.
        job_id = launched.get("job_id")
        if not isinstance(job_id, int):
            # No job id reached this process yet. A job the in-flight start
            # created finds no launch file and ends without running pytest.
            launch_path.unlink(missing_ok=True)
            return
        cancelled = _reap_job(job_id, reference=launched.get("reference"), env=client, launch_path=launch_path)
        if cancelled:
            on_exit()
        elif resource_state is not None:
            resource_state["preserve"] = True
            if preserve_guard is not None:
                preserve_guard()

    # Installed before the job exists: a signal between ``job start`` returning
    # its id and the wait would otherwise kill the waiter under the previous
    # handler and leave the queued job occupying the pytest pool.
    with _on_exit(reap_owned_job):
        try:
            started = _agentctl(
                [
                    "--json",
                    "job",
                    "start",
                    str(root),
                    PYTEST_OPERATION,
                    "--workspace",
                    str(root),
                    "--",
                    str(launch_path),
                ],
                env=client,
            )
            started_document = _document(started, verb="job start")
            job_id = started_document.get("job_id")
            reference = started_document.get("reference")
            reference = reference if isinstance(reference, str) and reference else None
            if not isinstance(job_id, int) or isinstance(job_id, bool):
                raise PytestSlotUnavailableError(REFUSAL.format(reason=f"`{AGENTCTL} job start` returned no job id"))
        except PytestSlotUnavailableError:
            launch_path.unlink(missing_ok=True)
            raise
        launched.update(job_id=job_id, reference=reference)
        # The child writes only to this log, never to the job's own stream, so
        # the path is named now: following it is how a queued run is watched.
        sys.stderr.write(
            f"  waiting for the host pytest slot ({AGENTCTL} job {job_id}, pool {PYTEST_POOL}); "
            f"pytest output: {log_path} ...\n"
        )
        sys.stderr.flush()
        view = _wait_for(job_id, reference=reference, env=client)
    receipt = _read_slot_result(log_path)
    try:
        returncode = _job_exit_status(view, receipt=receipt)
    except PytestSlotUnavailableError as exc:
        if view.get("terminal") is True:
            launch_path.unlink(missing_ok=True)
        exc.runtime_evidence = {
            "job_id": job_id,
            "phase": view.get("phase"),
            "exit_code": view.get("exit_code"),
            "pytest_slot_receipt": receipt,
            "pytest_slot_log": str(log_path),
        }
        raise
    launch_path.unlink(missing_ok=True)
    if returncode == 0:
        _telemetry_path(log_path).unlink(missing_ok=True)
    termination = _job_termination(reference) if returncode != 0 else None
    if termination is not None:
        sys.stderr.write(f"  {AGENTCTL} job {job_id} was ended by {termination['killer']} ({termination['unit']})\n")
    sys.stderr.write(f"  pytest slot released; output: {log_path}\n")
    sys.stderr.flush()
    return SlotOutcome(
        returncode=returncode,
        slot=f"{AGENTCTL} job {job_id}",
        log_path=log_path,
        receipt=receipt,
        termination=termination,
    )


def _slot_result_path(log_path: Path) -> Path:
    return log_path.with_suffix(".result.json")


def _read_slot_result(log_path: Path) -> dict[str, Any] | None:
    path = _slot_result_path(log_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _progress_counts(environment: Mapping[str, str]) -> dict[str, Any]:
    """Extract the last durable progress facts without making them authoritative."""
    counts: dict[str, Any] = {}
    selection_path = environment.get("POLYLOGUE_PYTEST_SELECTION_PATH")
    if selection_path:
        try:
            selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            selection = None
        if isinstance(selection, Mapping):
            for key in ("selected_count", "deselected_count"):
                value = selection.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    counts[key] = value
    events_path = environment.get("POLYLOGUE_PYTEST_EVENTS_PATH")
    if events_path:
        outcomes: dict[str, int] = {}
        completed: set[str] = set()
        try:
            lines = Path(events_path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, Mapping):
                continue
            if event.get("event") == "test_finished" and isinstance(event.get("nodeid"), str):
                completed.add(event["nodeid"])
                continue
            if event.get("event") != "test_report" or event.get("when") != "call":
                continue
            outcome = event.get("outcome")
            if isinstance(outcome, str):
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcomes:
            counts["outcomes"] = outcomes
            counts["terminal_count"] = len(completed)
        elif completed:
            counts["terminal_count"] = len(completed)
    return counts


class _ProgressSnapshot:
    """Incrementally read pytest's append-only event ledger for telemetry."""

    def __init__(self, environment: Mapping[str, str]) -> None:
        self._environment = environment
        self._offsets: dict[Path, int] = {}
        self._outcomes: dict[str, int] = {}
        self._completed: set[str] = set()
        self._bytes = 0
        self._mutex = threading.Lock()

    def __call__(self) -> dict[str, Any]:
        with self._mutex:
            counts: dict[str, Any] = {}
            selection_path = self._environment.get("POLYLOGUE_PYTEST_SELECTION_PATH")
            if selection_path:
                try:
                    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    selection = None
                if isinstance(selection, Mapping):
                    for key in ("selected_count", "deselected_count"):
                        value = selection.get(key)
                        if isinstance(value, int) and not isinstance(value, bool):
                            counts[key] = value
            raw_dir = self._environment.get("POLYLOGUE_PYTEST_EVENTS_DIR")
            raw_path = self._environment.get("POLYLOGUE_PYTEST_EVENTS_PATH")
            paths = sorted(Path(raw_dir).glob("*.jsonl")) if raw_dir else ([Path(raw_path)] if raw_path else [])
            for path in paths:
                try:
                    size = path.stat().st_size
                    offset = self._offsets.get(path, 0)
                    if size < offset:
                        offset = 0
                    with path.open("rb") as handle:
                        handle.seek(offset)
                        payload = handle.read()
                    complete, _, _partial = payload.rpartition(b"\n")
                    self._offsets[path] = offset + len(complete) + (1 if complete else 0)
                    self._bytes += len(complete)
                except OSError:
                    continue
                for line in complete.splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, Mapping):
                        continue
                    if event.get("event") == "test_finished" and isinstance(event.get("nodeid"), str):
                        self._completed.add(event["nodeid"])
                    elif event.get("event") == "test_report" and event.get("when") == "call":
                        outcome = event.get("outcome")
                        if isinstance(outcome, str):
                            self._outcomes[outcome] = self._outcomes.get(outcome, 0) + 1
            if self._outcomes:
                counts["outcomes"] = dict(self._outcomes)
                counts["terminal_count"] = len(self._completed)
            elif self._completed:
                counts["terminal_count"] = len(self._completed)
            counts["event_bytes"] = self._bytes
            return counts


def _telemetry_path(log_path: Path) -> Path:
    return log_path.with_suffix(".telemetry.json")


def _persist_telemetry_seed(
    path: Path,
    *,
    sizing: Mapping[str, Any] | None,
    progress: _ProgressSnapshot,
) -> None:
    """Publish sizing before pytest starts, so a kill before first sample is diagnosable."""
    document = {
        "schema_version": 1,
        "kind": "polylogue.pytest-slot-telemetry",
        "status": "starting",
        "sizing": dict(sizing) if sizing is not None else None,
        "progress": progress(),
        "memory": None,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError):
        pass


def _sizing_note(sizing: Mapping[str, Any] | None) -> str | None:
    """Why this run is narrower than it asked to be, or None when it is not."""
    if sizing is None or not sizing.get("narrowed"):
        return None
    bound = "the job cgroup" if sizing["basis"] == "cgroup_budget" else "the declared slice budget"
    return (
        f"pytest slot: {sizing['available_mib']} MiB from {bound} "
        f"(host {sizing['host_available_mib']} MiB, cgroup {sizing['cgroup_available_mib']} MiB) "
        f"holds {sizing['workers']} workers, not {sizing['requested_workers']}; "
        "running narrower rather than being killed."
    )


def _slot_receipt(
    *,
    status: str,
    elapsed_s: float,
    sizing: Mapping[str, Any] | None,
    memory: Mapping[str, Any] | None,
    exit_code: int | None = None,
    log_path: Path | None = None,
    extra: Mapping[str, Any] | None = None,
    profile: ChargeProfile | None = None,
) -> dict[str, Any]:
    """The run's durable result: how wide it ran, and what it took to run that wide.

    The width and the peak belong to the same document because neither answers
    the question alone: a peak is only over or under budget against the width
    that was chosen, and a width is only justified by what the run then took.

    ``corroboration`` closes that loop in the receipt itself. The sizing
    constants in ``worker_memory`` are single measurements; recording the
    verdict here means the next reader sees whether this run's sampler agreed
    with the profile it was admitted under, instead of the peak being written
    and never compared to anything (which is how ``WORKER_PEAK_CACHE_MIB``
    stayed at a back-solved residual across four runs that could have
    falsified it).
    """
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "polylogue.pytest-slot-result",
        "status": status,
        "elapsed_s": round(elapsed_s, 3),
    }
    if exit_code is not None:
        receipt["exit_code"] = exit_code
    if log_path is not None:
        receipt["log_path"] = str(log_path)
    if sizing is not None:
        receipt["sizing"] = dict(sizing)
    if memory is not None:
        receipt["memory"] = dict(memory)
    corroboration = (
        corroborate_profile(memory, sizing) if profile is None else corroborate_profile(memory, sizing, profile=profile)
    )
    if corroboration is not None:
        receipt["corroboration"] = corroboration
    if extra is not None:
        receipt.update(extra)
    return receipt


#: Asks the slot to identify, at the moment pytest starts, the worktree content
#: and branch it is about to run.
WORKTREE_PROVENANCE_ENV = "POLYLOGUE_FOCUSED_WORKTREE_PROVENANCE"


def _focused_worktree_provenance(cwd: str, environment: Mapping[str, str]) -> dict[str, Any] | None:
    if environment.get(WORKTREE_PROVENANCE_ENV) != "1":
        return None
    from devtools.checkout_identity import ALLOW_DEFAULT_BRANCH_ENV, checkout_identity, default_branch_refusal
    from devtools.verify_runs import git_dirty, git_worktree_content_sha256

    root = Path(cwd)
    # Identity, content, identity again: the three describe one checkout state
    # only if nothing moved between them.
    identity = checkout_identity(root)
    digest = git_worktree_content_sha256(root)
    dirty = git_dirty(root)
    if identity.head is None or digest is None:
        raise PytestSlotUnavailableError("focused worktree content could not be identified")
    if checkout_identity(root) != identity or git_worktree_content_sha256(root) != digest or git_dirty(root) != dirty:
        raise PytestSlotUnavailableError("the checkout changed while its content was being identified")
    head = identity.head
    # The branch admitted at submission may have changed while the run queued.
    refusal = default_branch_refusal(
        identity, command="devtools test", allowed=environment.get(ALLOW_DEFAULT_BRANCH_ENV) == "1"
    )
    if refusal is not None:
        raise PytestSlotUnavailableError(f"the checkout is on the default branch at slot start: {refusal}")
    return {
        "git_head": head,
        "git_branch": identity.branch,
        "git_dirty": dirty,
        "git_worktree_content_sha256": digest,
        "capture_source": "pytest_slot_start",
    }


def _persist_slot_result(log_path: Path, receipt: Mapping[str, Any]) -> None:
    """Atomically publish the result document the waiting client reads."""
    path = _slot_result_path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _write_interrupted_result(
    log_path: Path,
    *,
    environment: Mapping[str, str],
    started: float,
    signal_number: int,
    worktree_provenance: Mapping[str, Any] | None,
    sizing: Mapping[str, Any] | None = None,
    memory: Mapping[str, Any] | None = None,
    profile: ChargeProfile | None = None,
    execution_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically preserve an interruption result before the worker dies."""
    receipt = _slot_receipt(
        status="interrupted",
        elapsed_s=time.monotonic() - started,
        sizing=sizing,
        memory=memory,
        profile=profile,
        extra={
            **({"execution_source": dict(execution_source)} if execution_source is not None else {}),
            "diagnosis": "pytest_interrupted",
            "signal": signal.Signals(signal_number).name,
            **({"worktree_provenance": dict(worktree_provenance)} if worktree_provenance is not None else {}),
            "progress": _progress_counts(environment),
        },
    )
    _persist_slot_result(log_path, receipt)
    return receipt


def _report_admission_wait(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _run_held(
    argv: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    stdout: IO[Any] | None,
    on_exit: Callable[[], None],
    telemetry_path: Path | None = None,
    result_path: Path | None = None,
    on_interrupt: Callable[[], None] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run pytest here, in its own process group so a signalled waiter takes it along.

    The width is narrowed to what memory allows at this moment for the same
    reason the queued path narrows it inside the slot: this is where the run
    starts, and the corpus operation reaches pytest through here.

    ``result_path`` is where a terminated run's receipt is preserved. This
    path is the one ``verify_all``/``verify_affected`` take when they already
    hold the pytest pool, and it used to write nothing at all on SIGTERM: the
    handler stopped the child and re-raised, so neither ``sampler.stop()`` nor
    the return value was ever reached, and the caller's disposal removed the
    live telemetry sidecar on the way out. The selected width and the measured
    peak were lost on exactly the runs -- deadline kills -- whose sizing
    evidence is worth the most.
    """
    started = time.monotonic()
    worktree_provenance = _focused_worktree_provenance(cwd, env)
    profile, max_workers = charge_profile_for(env)
    ledger = admission_ledger(env)
    try:
        command, sizing = admit_width(
            argv,
            size=resize_worker_argument,
            profile=profile,
            max_workers=max_workers,
            ledger=ledger,
            report=_report_admission_wait,
        )
        if admission_not_ready(sizing):
            receipt = _slot_receipt(
                status="deferred",
                elapsed_s=time.monotonic() - started,
                sizing=sizing,
                memory=None,
                exit_code=EX_TEMPFAIL,
                profile=profile,
                extra={"diagnosis": RESOURCE_NOT_READY},
            )
            if result_path is not None:
                _persist_slot_result(result_path, receipt)
            return EX_TEMPFAIL, receipt
        return _run_held_admitted(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            on_exit=on_exit,
            started=started,
            worktree_provenance=worktree_provenance,
            profile=profile,
            sizing=sizing,
            telemetry_path=telemetry_path,
            result_path=result_path,
            on_interrupt=on_interrupt,
        )
    finally:
        if ledger is not None:
            ledger.release()


def _stop_execution_process(process: subprocess.Popen[Any], source_guard: Any = None) -> None:
    if source_guard is not None:
        source_guard.stop(process)
        return
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=STOP_TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=STOP_KILL_GRACE_S)


def _run_held_admitted(
    command: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    stdout: IO[Any] | None,
    on_exit: Callable[[], None],
    started: float,
    worktree_provenance: dict[str, Any] | None,
    profile: ChargeProfile,
    sizing: dict[str, Any] | None,
    telemetry_path: Path | None,
    result_path: Path | None,
    on_interrupt: Callable[[], None] | None,
) -> tuple[int, dict[str, Any]]:
    note = _sizing_note(sizing)
    if note is not None:
        sys.stderr.write(note + "\n")
        sys.stderr.flush()
    progress = _ProgressSnapshot(env)
    terminal_status = "running"
    if telemetry_path is not None:
        _persist_telemetry_seed(telemetry_path, sizing=sizing, progress=progress)
    if result_path is not None:
        # A previous run of this pid may have left one; reading that would
        # report someone else's interruption.
        _slot_result_path(result_path).unlink(missing_ok=True)
    from devtools.execution_source import complete_execution, finish_execution, start_execution

    guard = start_execution(Path(cwd), env)
    try:
        worktree_provenance = _focused_worktree_provenance(cwd, env)
        execution_command = guard.command(command, env, worktree_provenance) if guard is not None else command
        if guard is not None and worktree_provenance is not None:
            worktree_provenance["capture_source"] = "readonly_snapshot"
    except BaseException:
        if guard is not None:
            guard.failure = "execution setup failed"
        finish_execution(guard, env)
        raise
    child_environment = {**env, CUSTODY_ENV: uuid.uuid4().hex}
    try:
        process = subprocess.Popen(
            execution_command,
            cwd=cwd,
            env=child_environment,
            stdout=stdout,
            stderr=stdout,
            process_group=0,
            pass_fds=guard.pass_fds if guard is not None else (),
        )
    except BaseException:
        if guard is not None:
            guard.failure = "execution setup failed"
        finish_execution(guard, env)
        raise
    try:
        if guard is not None:
            guard.launched_process(process)
    except BaseException:
        _stop_execution_process(process, guard)
        finish_execution(guard, env)
        raise
    try:
        sampler = ProcessGroupMemorySampler(
            process.pid,
            custody_marker=child_environment[CUSTODY_ENV],
            snapshot_path=telemetry_path,
            snapshot_context=lambda: {
                "status": terminal_status,
                "pid": process.pid,
                "process_group": process.pid,
                "sizing": sizing,
                "progress": progress(),
            },
        )
    except BaseException:
        if guard is not None:
            guard.failure = "execution sampler setup failed"
        _stop_execution_process(process, guard)
        if guard is None:
            _group_reaped(process.pid)
        finish_execution(guard, env)
        raise
    try:
        sampler.start()
    except BaseException:
        if guard is not None:
            guard.failure = "execution sampler setup failed"
        _stop_execution_process(process, guard)
        with contextlib.suppress(Exception):
            sampler.stop()
        if guard is None:
            _group_reaped(process.pid)
        finish_execution(guard, env)
        raise

    def preserve(signal_number: int) -> None:
        """Write the terminated run's receipt before anything is disposed."""
        if guard is not None:
            guard.failure = "execution interrupted"
        if on_interrupt is not None:
            on_interrupt()
        if result_path is None:
            return
        # One reading taken here rather than trusting the sampling thread's
        # last one: this handler runs before ``stop()``, so the group is still
        # alive and this is the truest peak the run ever reaches. A run
        # signalled before the thread's first pass would otherwise persist a
        # receipt whose measurement is "no sample observed the process group".
        with contextlib.suppress(Exception):
            sampler.sample()
        with contextlib.suppress(OSError):
            _write_interrupted_result(
                result_path,
                environment=env,
                started=started,
                signal_number=signal_number,
                worktree_provenance=worktree_provenance if guard is None else None,
                execution_source={"status": "unavailable", "reason": "execution interrupted"}
                if guard is not None
                else None,
                sizing=sizing,
                profile=profile,
                memory=sampler.persist(),
            )

    def stop() -> None:
        _stop_execution_process(process, guard)

    returncode = 125
    try:
        # ``preserve`` runs before ``stop`` so the sampler reads the group
        # while it is still alive, and before ``on_exit`` so the caller's
        # disposal cannot remove the evidence first.
        with _on_exit(stop, on_exit, on_signal=preserve):
            returncode = process.wait()
            terminal_status = "passed" if returncode == 0 else "failed"
    finally:
        try:
            memory = sampler.stop()
        except BaseException:
            if guard is not None:
                guard.failure = "execution sampler settlement failed"
            raise
        finally:
            _stop_execution_process(process, guard)
            stability = finish_execution(guard, env, complete=complete_execution(command, env, returncode))
    if stability is not None and stability["status"] != "stable":
        returncode = 125
    return returncode, _slot_receipt(
        status="success" if returncode == 0 else "failed",
        profile=profile,
        exit_code=returncode,
        elapsed_s=time.monotonic() - started,
        sizing=sizing,
        memory=memory,
        extra={
            "worktree_provenance": worktree_provenance
            if stability is None or stability["status"] == "stable"
            else None,
            "execution_source": stability,
            **(
                {"diagnosis": "execution_source_unavailable"}
                if stability is not None and stability["status"] != "stable"
                else {}
            ),
        },
    )


def run_pytest(
    command: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    root: Path,
    stdout: IO[Any] | None = None,
) -> SlotOutcome:
    """Run a managed pytest command, acquiring the host's pytest slot first.

    A caller inside the pytest pool (or marked ``POLYLOGUE_PYTEST_SLOT=held``)
    runs the command here, streaming as before. Every other caller submits it
    as the ``pytest_focused`` operation in the host's single-slot pytest pool
    and reads the captured log.

    Successful runs remove both temporary trees. Failed, interrupted, or
    possibly still queued runs retain the resources needed for diagnosis or
    execution.
    """
    argv, contained, scratch = contained_pytest_run(command, env=env, root=root)
    basetemp = scratch.with_name(scratch.name.removesuffix(".tmpdir"))
    telemetry_path = scratch.parent / f"telemetry-{scratch.name}.json"
    # A terminated held run writes its receipt beside the queued path's, under
    # the checkout's retained artifact directory, not into the scratch tree
    # ``dispose`` removes.
    held_result_path = root / LAUNCH_DIR / f"pytest-slot-held-{os.getpid()}.log"
    sweep_stale_temp_trees(basetemp.parent)
    guard = guard_temp_trees(scratch, basetemp)
    keep = False
    resource_state = {"preserve": False}

    def dispose() -> None:
        guard.cancel()
        if not keep and not resource_state["preserve"]:
            remove_temp_tree(scratch)
            remove_temp_tree(basetemp)
            telemetry_path.unlink(missing_ok=True)

    def mark_interrupted() -> None:
        nonlocal keep
        keep = True

    try:
        if holds_pytest_slot(env):
            returncode, receipt = _run_held(
                argv,
                cwd=cwd,
                env=contained,
                stdout=stdout,
                on_exit=dispose,
                telemetry_path=telemetry_path,
                result_path=held_result_path,
                on_interrupt=mark_interrupted,
            )
            outcome = SlotOutcome(returncode=returncode, slot=SLOT_HELD, receipt=receipt)
        else:
            outcome = _submit(
                argv,
                cwd=cwd,
                env=contained,
                root=root,
                on_exit=dispose,
                resource_state=resource_state,
                preserve_guard=guard.cancel,
            )
        # A separate diagnostic cannot clear the original failed launch.
        keep = outcome.returncode != 0
        return outcome
    finally:
        dispose()


def run_pytest_isolated(
    command: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    root: Path,
    stdout: IO[Any] | None = None,
) -> SlotOutcome:
    """Run through the managed containment/receipt harness without agentctl.

    CI has no local AgentCTL runtime. It must opt into this mode explicitly;
    workstation callers use :func:`run_pytest` and fail closed if admission is
    unavailable.
    """
    argv, contained, scratch = contained_pytest_run(command, env=env, root=root)
    basetemp = scratch.with_name(scratch.name.removesuffix(".tmpdir"))
    telemetry_path = scratch.parent / f"telemetry-{scratch.name}.json"
    held_result_path = root / LAUNCH_DIR / f"pytest-slot-held-{os.getpid()}.log"
    sweep_stale_temp_trees(basetemp.parent)
    guard = guard_temp_trees(scratch, basetemp)
    keep = False
    resource_state = {"preserve": False}

    def dispose() -> None:
        guard.cancel()
        if not keep and not resource_state["preserve"]:
            remove_temp_tree(scratch)
            remove_temp_tree(basetemp)
            telemetry_path.unlink(missing_ok=True)

    def mark_interrupted() -> None:
        nonlocal keep
        keep = True

    try:
        returncode, receipt = _run_held(
            argv,
            cwd=cwd,
            env=contained,
            stdout=stdout,
            on_exit=dispose,
            telemetry_path=telemetry_path,
            result_path=held_result_path,
            on_interrupt=mark_interrupted,
        )
        keep = returncode != 0
        return SlotOutcome(returncode=returncode, slot="isolated", receipt=receipt)
    finally:
        dispose()


def _read_launch(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("launch file must be an object")
    for key in ("argv", "working_directory", "environment", "log_path"):
        if key not in document:
            raise ValueError(f"launch file is missing {key!r}")
    return document


def _group_alive(pgid: int) -> bool:
    """Whether any process of group ``pgid`` still runs.

    A zombie is not running: it holds no slot state and is waiting only on a
    reaper this launch does not own (a subreaper that never waits can keep
    one for the life of the job). ``killpg(pgid, 0)`` succeeds on a
    zombie-only group, so where ``/proc`` exists, members are read from it.
    """
    proc = Path("/proc")
    if not proc.is_dir():
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_line = (entry / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # ``pid (comm) state ppid pgrp ...``; comm may itself contain ")".
        fields = stat_line[stat_line.rfind(")") + 2 :].split()
        if len(fields) >= 3 and fields[2] == str(pgid) and fields[0] not in {"Z", "X"}:
            return True
    return False


def _group_reaped(pgid: int) -> bool:
    """Terminate what is left of a process group; whether it is gone.

    The group leader is already reaped, so what remains are its descendants,
    which are waited on by polling the group rather than by ``wait``.
    """
    for sig, grace in ((signal.SIGTERM, STOP_TERM_GRACE_S), (signal.SIGKILL, STOP_KILL_GRACE_S)):
        if not _group_alive(pgid):
            return True
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, sig)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and _group_alive(pgid):
            time.sleep(0.05)
    return not _group_alive(pgid)


def _run_launch(launch_path: Path) -> int:
    """Run one launch file inside the job holding the pytest slot."""
    try:
        launch = _read_launch(launch_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"devtools.pytest_slot: unusable launch file: {exc}\n")
        return 2
    # The launch file carries the resolved environment; it must not outlive the
    # run that consumes it.
    launch_path.unlink(missing_ok=True)
    environment = dict(launch["environment"])
    ISOLATED_ARCHIVE_PARENT.mkdir(parents=True, exist_ok=True)
    archive_root = Path(tempfile.mkdtemp(prefix="pytest-", dir=ISOLATED_ARCHIVE_PARENT))
    environment["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    environment[SLOT_ESCAPE_ENV] = SLOT_HELD
    worktree_provenance = None
    log_path = Path(launch["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child: subprocess.Popen[Any] | None = None
    sampler: ProcessGroupMemorySampler | None = None
    sizing: dict[str, Any] | None = None
    progress = _ProgressSnapshot(environment)
    telemetry_path = _telemetry_path(log_path)
    started = time.monotonic()
    terminating = False
    terminal_status = "running"

    # Chosen before the signal handler exists: an interrupted receipt must
    # corroborate against the same profile the width was admitted under.
    profile, max_workers = charge_profile_for(environment)

    #: Process groups this launch started, reaped by the signal handler.
    started_groups: list[int] = []

    def terminate_on_signal(signal_number: int, _frame: object) -> None:
        nonlocal terminating
        if terminating:
            return
        terminating = True
        if guard is not None:
            guard.failure = "execution interrupted"
        if child is not None:
            _stop_execution_process(child, guard)
        if guard is None:
            for group in started_groups:
                _group_reaped(group)
        stability = finish_execution(guard, environment)
        with contextlib.suppress(OSError):
            receipt = _write_interrupted_result(
                log_path,
                environment=environment,
                started=started,
                signal_number=signal_number,
                worktree_provenance=worktree_provenance if guard is None else None,
                execution_source=stability,
                sizing=sizing,
                profile=profile,
                memory=sampler.persist() if sampler is not None else None,
            )
            _print_result(receipt)
        os._exit(128 + signal_number)

    guard = None
    previous = {number: signal.signal(number, terminate_on_signal) for number in REAPED_SIGNALS}
    ledger = None
    from devtools.execution_source import complete_execution, finish_execution, start_execution

    try:
        ledger = admission_ledger(environment)
        command, sizing = admit_width(
            launch["argv"],
            size=resize_worker_argument,
            profile=profile,
            max_workers=max_workers,
            ledger=ledger,
            report=_report_admission_wait,
        )
        if admission_not_ready(sizing):
            receipt = _slot_receipt(
                status="deferred",
                elapsed_s=time.monotonic() - started,
                sizing=sizing,
                memory=None,
                exit_code=EX_TEMPFAIL,
                log_path=log_path,
                profile=profile,
                extra={"diagnosis": RESOURCE_NOT_READY},
            )
            with open(log_path, "wb") as log:
                log.write((json.dumps(receipt, sort_keys=True) + "\n").encode())
            _persist_slot_result(log_path, receipt)
            _print_result(receipt)
            return EX_TEMPFAIL
        _persist_telemetry_seed(telemetry_path, sizing=sizing, progress=progress)
        note = _sizing_note(sizing)
        with open(log_path, "wb") as log:
            if note is not None:
                log.write((note + "\n").encode())
                log.flush()
            try:
                guard = start_execution(Path(launch["working_directory"]), environment)
                worktree_provenance = _focused_worktree_provenance(launch["working_directory"], environment)
                execution_command = (
                    guard.command(command, environment, worktree_provenance) if guard is not None else command
                )
                if guard is not None and worktree_provenance is not None:
                    worktree_provenance["capture_source"] = "readonly_snapshot"
                environment[CUSTODY_ENV] = uuid.uuid4().hex
                child = subprocess.Popen(
                    execution_command,
                    pass_fds=guard.pass_fds if guard is not None else (),
                    cwd=launch["working_directory"],
                    env=environment,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                # Live telemetry remains bound to this original attempt.
                if guard is not None:
                    guard.launched_process(child)
                measured = [child]
                started_groups.append(child.pid)
                sampler = ProcessGroupMemorySampler(
                    child.pid,
                    custody_marker=environment[CUSTODY_ENV],
                    snapshot_path=telemetry_path,
                    snapshot_context=lambda: {
                        "status": terminal_status,
                        "pid": measured[0].pid,
                        "process_group": measured[0].pid,
                        "sizing": sizing,
                        "progress": progress(),
                    },
                )
                sampler.start()
                returncode = child.wait()
                terminal_status = "passed" if returncode == 0 else "failed"
            except (OSError, PytestSlotUnavailableError) as exc:
                log.write(f"devtools.pytest_slot: could not start pytest: {exc}\n".encode())
                if guard is not None:
                    guard.failure = str(exc)
                returncode = 125
            finally:
                if child is not None:
                    _stop_execution_process(child, guard)
                memory = sampler.stop() if sampler is not None else None
        if guard is None:
            for group in started_groups:
                _group_reaped(group)
        stability = finish_execution(guard, environment, complete=complete_execution(command, environment, returncode))
        guard = None
        if stability is not None and stability["status"] != "stable":
            returncode = 125
        receipt = _slot_receipt(
            status="success" if returncode == 0 else "failed",
            profile=profile,
            exit_code=returncode,
            elapsed_s=time.monotonic() - started,
            sizing=sizing,
            memory=memory,
            log_path=log_path,
            extra={
                "worktree_provenance": worktree_provenance
                if stability is None or stability["status"] == "stable"
                else None,
                "execution_source": stability,
                **(
                    {"diagnosis": "execution_source_unavailable"}
                    if stability is not None and stability["status"] != "stable"
                    else {}
                ),
            },
        )
        # Written as well as printed: the waiting client reads the file, and the
        # job's stdout is the result artifact.
        with contextlib.suppress(OSError):
            _persist_slot_result(log_path, receipt)
        _print_result(receipt)
        return returncode

    finally:
        if guard is not None:
            guard.failure = guard.failure or "execution did not finish"
            if child is not None:
                _stop_execution_process(child, guard)
        finish_execution(guard, environment)
        if ledger is not None:
            ledger.release()
        if archive_root.parent == ISOLATED_ARCHIVE_PARENT:
            remove_temp_tree(archive_root)
        for number, handler in previous.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(number, handler)


def _print_result(document: Mapping[str, Any]) -> None:
    """The job's typed result: its stdout is the result artifact."""
    with contextlib.suppress(OSError):
        sys.stdout.write(json.dumps(document, sort_keys=True) + "\n")
        sys.stdout.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """The ``pytest_focused`` operation: one launch file, or a ``devtools test`` selection.

    Either way the job is in the pytest pool, so the run holds the slot and
    executes in place.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) == 1 and arguments[0].endswith(".json"):
        return _run_launch(Path(arguments[0]))
    if not arguments:
        sys.stderr.write("usage: python -m devtools.pytest_slot <launch file> | <devtools test selection>\n")
        return 2
    from devtools.run_tests import main as run_focused_tests

    ISOLATED_ARCHIVE_PARENT.mkdir(parents=True, exist_ok=True)
    archive_root = Path(tempfile.mkdtemp(prefix="pytest-", dir=ISOLATED_ARCHIVE_PARENT))
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    try:
        return run_focused_tests(arguments)
    finally:
        remove_temp_tree(archive_root)


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())

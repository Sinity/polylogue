"""The host's single pytest slot, acquired through pueue.

pytest is the heaviest thing this checkout runs, and several agent sessions
share one workstation. A run started from a session subagent sits outside every
load control the daemon applies to its own jobs, so concurrent runs contend for
the same cores and disk until a long job passes its timeout.

Every managed pytest run therefore holds the host's `pytest` pueue group (one
task at a time). A run already inside the pytest queue task holds the slot
already, marked explicitly with ``POLYLOGUE_PYTEST_SLOT=held``. Generic
Sinnixd job identity does not imply pytest-slot ownership: lane jobs queue and
wait here.

pueue 4 records the full client environment of ``pueue add`` into a user-only
state file, so the adder runs with a reduced environment and the managed pytest
environment travels in the launch file instead.

Every run started here also keeps its temporary trees inside the checkout; see
:func:`basetemp_root`.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final

from devtools.agent_env import inside_declared_pytest_worker
from devtools.cloud_sentinels import cloud_sentinel_declined

__all__ = [
    "BASETEMP_ROOT_ENV",
    "INHERITED_ENVIRONMENT_KEYS",
    "REAPED_SIGNALS",
    "PYTEST_GROUP",
    "PytestSlotUnavailableError",
    "SlotOutcome",
    "basetemp_root",
    "contained_pytest_run",
    "holds_pytest_slot",
    "main",
    "remove_temp_tree",
    "run_pytest",
]

#: Exported into every task started by ``sinnixd-queue-run``.
QUEUE_TASK_ENV: Final = "AGENTCTL_JOB_ID"
#: The installed queue runner moves the workload out of pueued.service and
#: into the slice selected by the launch document's pool.
#: The queue runner, newest name first; the daemon-era name still resolves on
#: a host that has not switched.
QUEUE_RUNNERS: Final = ("agentctl-run", "sinnixd-queue-run")
QUEUE_RUNNER: Final = QUEUE_RUNNERS[0]
#: Explicit escape, for the hermetic test of this mechanism.
SLOT_ESCAPE_ENV: Final = "POLYLOGUE_PYTEST_SLOT"
SLOT_HELD: Final = "held"

#: The host group whose parallelism is one.
PYTEST_GROUP: Final = "pytest"

#: Signals that end this process while it waits on a task it owns. The task
#: outlives the waiter, and the slot's parallelism is one, so an unreaped task
#: starves every other checkout on the host until someone notices.
REAPED_SIGNALS: Final[tuple[signal.Signals, ...]] = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

#: The only keys the ``pueue add`` client inherits. Everything pytest needs
#: travels in the launch file, because pueue persists the adder's environment
#: into shared state.
INHERITED_ENVIRONMENT_KEYS: Final[tuple[str, ...]] = ("HOME", "PATH", "XDG_RUNTIME_DIR", "XDG_DATA_HOME")

LAUNCH_DIR: Final = Path(".cache/verify")

REFUSAL = (
    "pytest could not acquire the host's pytest slot: {reason}. "
    "Start the queue with `systemctl --user start pueued`, then rerun. "
    "Set POLYLOGUE_PYTEST_SLOT=held only when the caller already holds the slot."
)


class PytestSlotUnavailableError(RuntimeError):
    """The pytest slot could not be acquired, and the run must not proceed."""


@dataclass(frozen=True)
class SlotOutcome:
    returncode: int
    #: What the receipt records: ``pueue task 12`` or ``held``.
    slot: str
    #: Where the queued run's output landed, or None when it streamed.
    log_path: Path | None = None
    receipt: dict[str, Any] | None = None


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
    rmtree cannot unlink them and silently leaves the tree behind.
    """
    for parent, directories, files in os.walk(path, topdown=False):
        for name in (*directories, *files):
            with contextlib.suppress(OSError):
                os.chmod(os.path.join(parent, name), 0o700)
    shutil.rmtree(path, ignore_errors=True)


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
    for. Returns the scratch directory the caller owns.
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


def holds_pytest_slot(env: Mapping[str, str]) -> bool:
    """Whether this process is already inside the host's pytest slot."""
    return env.get(SLOT_ESCAPE_ENV) == SLOT_HELD or inside_declared_pytest_worker(env)


def adder_environment(env: Mapping[str, str]) -> dict[str, str]:
    """The reduced environment the ``pueue add`` client runs with."""
    return {key: env[key] for key in INHERITED_ENVIRONMENT_KEYS if env.get(key)}


def _pueue_executable(env: Mapping[str, str]) -> str:
    found = shutil.which("pueue", path=env.get("PATH") or os.defpath)
    if found is None:
        raise PytestSlotUnavailableError(REFUSAL.format(reason="`pueue` is not on PATH"))
    return found


def _pueue(arguments: Sequence[str], *, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    executable = _pueue_executable(env)
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
            REFUSAL.format(reason=f"`pueue {arguments[0]}` could not start: {exc}")
        ) from exc


def _cancel_task(task_id: str, *, env: Mapping[str, str]) -> None:
    """Cancel through the owner that also empties the task's systemd scope."""
    executable = shutil.which("agentctl", path=env.get("PATH") or os.defpath)
    if executable is None:
        raise PytestSlotUnavailableError(REFUSAL.format(reason="`agentctl` is not on PATH"))
    try:
        completed = subprocess.run(
            [executable, "job", "cancel", task_id],
            env=dict(env),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise PytestSlotUnavailableError(
            REFUSAL.format(reason=f"`agentctl job cancel` could not start: {exc}")
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PytestSlotUnavailableError(REFUSAL.format(reason=f"`agentctl job cancel` failed: {detail}"))


def _task_result(status_json: str, task_id: str) -> int:
    try:
        document = json.loads(status_json)
    except json.JSONDecodeError as exc:
        raise PytestSlotUnavailableError(REFUSAL.format(reason=f"`pueue status --json` was unreadable: {exc}")) from exc
    task = (document.get("tasks") or {}).get(task_id) if isinstance(document, Mapping) else None
    if not isinstance(task, Mapping):
        raise PytestSlotUnavailableError(REFUSAL.format(reason=f"pueue no longer knows task {task_id}"))
    status = task.get("status")
    detail = status.get("Done") if isinstance(status, Mapping) else None
    if not isinstance(detail, Mapping):
        raise PytestSlotUnavailableError(REFUSAL.format(reason=f"pueue task {task_id} did not reach Done: {status!r}"))
    result = detail.get("result")
    if result == "Success":
        return 0
    if isinstance(result, Mapping) and "Failed" in result:
        return int(result["Failed"])
    # Killed, and every other terminal-but-not-exited result.
    raise PytestSlotUnavailableError(REFUSAL.format(reason=f"pueue task {task_id} ended as {result!r}"))


def _reap_task(task_id: str, *, env: Mapping[str, str], launch_path: Path | None = None) -> None:
    """End a task this process owns and empty its execution scope.

    Best effort by construction: the reason we are here is that the waiter is
    being killed, so a failing reap must not replace the original cause of
    death with its own error. AgentCTL owns cancellation because pueue kills
    the queue runner with SIGKILL, which leaves the workload alive in the
    transient systemd scope that only AgentCTL names and stops.

    The launch file belongs to whoever ends the run, so deleting it is part of
    a cancellation that succeeded: a task still on the queue reads it when it
    starts.
    """

    cancelled = False
    try:
        _cancel_task(task_id, env=env)
        cancelled = True
    except PytestSlotUnavailableError:
        pass
    if cancelled and launch_path is not None:
        with contextlib.suppress(OSError):
            launch_path.unlink(missing_ok=True)


@contextlib.contextmanager
def _reaping(task_id: str, *, env: Mapping[str, str], launch_path: Path | None = None) -> Iterator[None]:
    """Reap ``task_id`` if this process is signalled or unwound while waiting."""

    def handle(signal_number: int, frame: object) -> None:
        _reap_task(task_id, env=env, launch_path=launch_path)
        signal.signal(signal_number, previous.get(signal.Signals(signal_number), signal.SIG_DFL))
        os.kill(os.getpid(), signal_number)

    previous: dict[signal.Signals, Any] = {}
    for number in REAPED_SIGNALS:
        with contextlib.suppress(ValueError, OSError):
            previous[number] = signal.signal(number, handle)
    try:
        yield
    except BaseException:
        _reap_task(task_id, env=env, launch_path=launch_path)
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
    job_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "job_id": job_id,
        "project_id": "polylogue",
        "operation": "test",
        "pool": PYTEST_GROUP,
        "argv": list(argv),
        "working_directory": cwd,
        "environment": dict(env),
        "timeout_seconds": 3600,
        "result_kind": "exit",
        "log_path": str(log_path),
    }
    # The launch file carries the resolved environment; keep it off other users.
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump(document, stream)


def _queue(
    command: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    root: Path,
    label: str,
) -> SlotOutcome:
    identity = f"{os.getpid()}"
    launch_path = root / LAUNCH_DIR / f"pytest-slot-{identity}.json"
    log_path = root / LAUNCH_DIR / f"pytest-slot-{identity}.log"
    adder = adder_environment(env)
    queue_runner = next(
        (found for name in QUEUE_RUNNERS if (found := shutil.which(name, path=adder.get("PATH") or os.defpath))),
        None,
    )
    if queue_runner is None:
        raise PytestSlotUnavailableError(REFUSAL.format(reason=f"`{QUEUE_RUNNER}` is not on PATH"))
    _write_launch(
        launch_path,
        argv=command,
        cwd=cwd,
        env=env,
        log_path=log_path,
        job_id=f"polylogue-test-{identity}",
    )
    added = _pueue(
        [
            "add",
            "--escape",
            "--group",
            PYTEST_GROUP,
            "--label",
            label,
            "--working-directory",
            str(root),
            "--print-task-id",
            "--",
            queue_runner,
            str(launch_path),
        ],
        env=adder,
    )
    if added.returncode != 0:
        launch_path.unlink(missing_ok=True)
        raise PytestSlotUnavailableError(
            REFUSAL.format(reason=f"`pueue add` failed: {added.stderr.strip() or added.stdout.strip()}")
        )
    task_id = added.stdout.strip()
    if not task_id.isdigit():
        launch_path.unlink(missing_ok=True)
        raise PytestSlotUnavailableError(REFUSAL.format(reason=f"`pueue add` printed no task id: {added.stdout!r}"))
    sys.stderr.write(f"  waiting for the host pytest slot (pueue task {task_id}, group {PYTEST_GROUP}) ...\n")
    sys.stderr.flush()
    with _reaping(task_id, env=adder, launch_path=launch_path):
        _pueue(["wait", task_id], env=adder)
        status = _pueue(["status", "--json"], env=adder)
        receipt = _read_timeout_receipt(log_path)
        try:
            returncode = _task_result(status.stdout, task_id)
        except PytestSlotUnavailableError:
            if receipt is None or receipt.get("status") != "timed_out":
                raise
            returncode = 124
    launch_path.unlink(missing_ok=True)
    sys.stderr.write(f"  pytest slot released; output: {log_path}\n")
    sys.stderr.flush()
    return SlotOutcome(
        returncode=returncode,
        slot=f"pueue task {task_id}",
        log_path=log_path,
        receipt=receipt,
    )


def _timeout_receipt_path(log_path: Path) -> Path:
    return log_path.with_suffix(".result.json")


def _read_timeout_receipt(log_path: Path) -> dict[str, Any] | None:
    path = _timeout_receipt_path(log_path)
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
        try:
            lines = Path(events_path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, Mapping) or event.get("event") != "test_report":
                continue
            outcome = event.get("outcome")
            if isinstance(outcome, str):
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcomes:
            counts["outcomes"] = outcomes
            counts["terminal_count"] = sum(outcomes.values())
    return counts


def _write_timeout_receipt(
    log_path: Path, *, environment: Mapping[str, str], started: float, signal_number: int
) -> dict[str, Any]:
    """Atomically preserve a typed timeout result before the worker dies."""
    receipt = {
        "schema_version": 1,
        "kind": "polylogue.pytest-slot-result",
        "status": "timed_out",
        "diagnosis": "pytest_deadline",
        "signal": signal.Signals(signal_number).name,
        "elapsed_s": round(time.monotonic() - started, 3),
        "progress": _progress_counts(environment),
    }
    path = _timeout_receipt_path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return receipt


def run_pytest(
    command: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    root: Path,
    label: str,
    stdout: IO[Any] | None = None,
) -> SlotOutcome:
    """Run a managed pytest command, acquiring the host's pytest slot first.

    When the pytest-group runner sets ``POLYLOGUE_PYTEST_SLOT=held``, the slot
    is already held and the command runs here, streaming as before. Every other
    caller is queued in the host's single-slot ``pytest`` group and its output
    is captured.
    """
    argv, contained, scratch = contained_pytest_run(command, env=env, root=root)
    if holds_pytest_slot(env):
        completed = subprocess.run(argv, cwd=cwd, env=contained, stdout=stdout, stderr=stdout)
        outcome = SlotOutcome(returncode=completed.returncode, slot=SLOT_HELD)
    else:
        outcome = _queue(argv, cwd=cwd, env=contained, root=root, label=label)
    # A failed run keeps its scratch: that is when the leftovers are worth
    # reading.
    if outcome.returncode == 0:
        remove_temp_tree(scratch)
    return outcome


def _read_launch(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("launch file must be an object")
    for key in ("argv", "working_directory", "environment", "log_path"):
        if key not in document:
            raise ValueError(f"launch file is missing {key!r}")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    """Run one launch file inside the pueue task holding the pytest slot."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        sys.stderr.write("usage: python -m devtools.pytest_slot <launch file>\n")
        return 2
    launch_path = Path(arguments[0])
    try:
        launch = _read_launch(launch_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"devtools.pytest_slot: unusable launch file: {exc}\n")
        return 2
    # The launch file carries the resolved environment; it must not outlive the
    # run that consumes it.
    launch_path.unlink(missing_ok=True)
    environment = dict(launch["environment"])
    environment[SLOT_ESCAPE_ENV] = SLOT_HELD
    log_path = Path(launch["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child: subprocess.Popen[Any] | None = None
    started = time.monotonic()
    terminating = False

    def terminate_on_signal(signal_number: int, _frame: object) -> None:
        nonlocal terminating
        if terminating:
            return
        terminating = True
        if child is not None and child.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(child.pid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    child.wait(timeout=2)
        with contextlib.suppress(OSError):
            _write_timeout_receipt(
                log_path,
                environment=environment,
                started=started,
                signal_number=signal_number,
            )
        os._exit(128 + signal_number)

    previous = {number: signal.signal(number, terminate_on_signal) for number in REAPED_SIGNALS}
    with open(log_path, "wb") as log:
        try:
            child = subprocess.Popen(
                list(launch["argv"]),
                cwd=launch["working_directory"],
                env=environment,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            returncode = child.wait()
        except OSError as exc:
            log.write(f"devtools.pytest_slot: could not start pytest: {exc}\n".encode())
            return 125
        finally:
            for number, handler in previous.items():
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(number, handler)
    return returncode


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())

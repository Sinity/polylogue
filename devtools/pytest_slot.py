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
disposes of them on every exit but a failed run's; see :func:`basetemp_root`.
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
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final

from devtools.agent_env import PYTEST_POOL, inside_pytest_pool
from devtools.cloud_sentinels import cloud_sentinel_declined

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
    "holds_pytest_slot",
    "main",
    "remove_temp_tree",
    "run_pytest",
]

#: The runtime's command line.
AGENTCTL: Final = "agentctl"
#: The declared operation in the pytest pool that runs one launch file.
PYTEST_OPERATION: Final = "pytest_focused"
#: Explicit escape, for the hermetic test of this mechanism.
SLOT_ESCAPE_ENV: Final = "POLYLOGUE_PYTEST_SLOT"
SLOT_HELD: Final = "held"
#: How often the waiter asks the runtime whether its job has ended.
POLL_INTERVAL_S: Final = 2.0

#: Signals that end this process while it waits on a job it owns. The job
#: outlives the waiter, and the pool's parallelism is one, so an unreaped job
#: starves every other checkout on the host until someone notices.
REAPED_SIGNALS: Final[tuple[signal.Signals, ...]] = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

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
    #: What the receipt records: ``agentctl job 12`` or ``held``.
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


def holds_pytest_slot(env: Mapping[str, str], *, cgroup_reader: Callable[[], str] | None = None) -> bool:
    """Whether this process is already inside the host's pytest slot.

    Ownership is the pytest pool: the cgroup the runtime placed the job in, or
    the pool name it exported. A job id alone never is.
    """
    return env.get(SLOT_ESCAPE_ENV) == SLOT_HELD or inside_pytest_pool(env, cgroup_reader=cgroup_reader)


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


def _cancel_job(job_id: int, *, env: Mapping[str, str]) -> None:
    """Cancel through the owner that also stops the job's transient unit."""
    completed = _agentctl(["job", "cancel", str(job_id)], env=env)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PytestSlotUnavailableError(REFUSAL.format(reason=f"`{AGENTCTL} job cancel` failed: {detail}"))


def _wait_for(job_id: int, *, env: Mapping[str, str]) -> dict[str, Any]:
    """The job's terminal view. The queue wait has no deadline; the job itself has one."""
    while True:
        view = _document(_agentctl(["--json", "job", "get", str(job_id)], env=env), verb="job get")
        if view.get("terminal"):
            return view
        time.sleep(POLL_INTERVAL_S)


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


def _reap_job(job_id: int, *, env: Mapping[str, str], launch_path: Path | None = None) -> None:
    """End a job this process owns and stop its transient unit.

    Best effort by construction: the reason we are here is that the waiter is
    being killed, so a failing reap must not replace the original cause of
    death with its own error.

    The launch file belongs to whoever ends the run, so deleting it is part of
    a cancellation that succeeded: a job still on the queue reads it when it
    starts.
    """
    cancelled = False
    try:
        _cancel_job(job_id, env=env)
        cancelled = True
    except PytestSlotUnavailableError:
        pass
    if cancelled and launch_path is not None:
        with contextlib.suppress(OSError):
            launch_path.unlink(missing_ok=True)


@contextlib.contextmanager
def _on_exit(*actions: Callable[[], None]) -> Iterator[None]:
    """Run ``actions`` if this process is signalled or unwound inside the block.

    A signal runs the actions, restores the previous handler and re-raises the
    signal, so an outer handler (or the default action) decides what the
    signal means; the actions are what must not be skipped on the way there.
    """

    def run_actions() -> None:
        for action in actions:
            with contextlib.suppress(Exception):
                action()

    def handle(signal_number: int, frame: object) -> None:
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
    document = {
        "kind": "polylogue.pytest-slot-launch",
        "argv": list(argv),
        "working_directory": cwd,
        "environment": dict(env),
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
) -> SlotOutcome:
    """Run ``command`` as the declared pytest-pool operation and wait for it."""
    identity = f"{os.getpid()}"
    launch_path = root / LAUNCH_DIR / f"pytest-slot-{identity}.json"
    log_path = root / LAUNCH_DIR / f"pytest-slot-{identity}.log"
    client = client_environment(env)
    _write_launch(launch_path, argv=command, cwd=cwd, env=env, log_path=log_path)
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
        job_id = _document(started, verb="job start").get("job_id")
        if not isinstance(job_id, int) or isinstance(job_id, bool):
            raise PytestSlotUnavailableError(REFUSAL.format(reason=f"`{AGENTCTL} job start` returned no job id"))
    except PytestSlotUnavailableError:
        launch_path.unlink(missing_ok=True)
        raise
    sys.stderr.write(f"  waiting for the host pytest slot ({AGENTCTL} job {job_id}, pool {PYTEST_POOL}) ...\n")
    sys.stderr.flush()
    with _on_exit(lambda: _reap_job(job_id, env=client, launch_path=launch_path), on_exit):
        view = _wait_for(job_id, env=client)
    receipt = _read_timeout_receipt(log_path)
    returncode = _job_exit_status(view, receipt=receipt)
    launch_path.unlink(missing_ok=True)
    sys.stderr.write(f"  pytest slot released; output: {log_path}\n")
    sys.stderr.flush()
    return SlotOutcome(
        returncode=returncode,
        slot=f"{AGENTCTL} job {job_id}",
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


def _run_held(
    argv: Sequence[str], *, cwd: str, env: Mapping[str, str], stdout: IO[Any] | None, on_exit: Callable[[], None]
) -> int:
    """Run pytest here, in its own process group so a signalled waiter takes it along."""
    process = subprocess.Popen(list(argv), cwd=cwd, env=dict(env), stdout=stdout, stderr=stdout, process_group=0)

    def stop() -> None:
        if process.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)

    with _on_exit(stop, on_exit):
        return process.wait()


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

    The temporary trees are removed on every exit except a failed run's, whose
    leftovers are worth reading.
    """
    argv, contained, scratch = contained_pytest_run(command, env=env, root=root)
    basetemp = scratch.with_name(scratch.name.removesuffix(".tmpdir"))
    keep = False

    def dispose() -> None:
        if not keep:
            remove_temp_tree(scratch)
            remove_temp_tree(basetemp)

    try:
        if holds_pytest_slot(env):
            returncode = _run_held(argv, cwd=cwd, env=contained, stdout=stdout, on_exit=dispose)
            outcome = SlotOutcome(returncode=returncode, slot=SLOT_HELD)
        else:
            outcome = _submit(argv, cwd=cwd, env=contained, root=root, on_exit=dispose)
        keep = outcome.returncode != 0
        return outcome
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
            receipt = _write_timeout_receipt(
                log_path,
                environment=environment,
                started=started,
                signal_number=signal_number,
            )
            _print_result(receipt)
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
    _print_result(
        {
            "schema_version": 1,
            "kind": "polylogue.pytest-slot-result",
            "status": "success" if returncode == 0 else "failed",
            "exit_code": returncode,
            "elapsed_s": round(time.monotonic() - started, 3),
            "log_path": str(log_path),
        }
    )
    return returncode


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

    return run_focused_tests(arguments)


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())

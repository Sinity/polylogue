"""The host's single pytest slot, acquired through pueue.

pytest is the heaviest thing this checkout runs, and several agent sessions
share one workstation. A run started from a session subagent sits outside every
load control the daemon applies to its own jobs, so concurrent runs contend for
the same cores and disk until a long job passes its timeout.

Every managed pytest run therefore holds the host's `pytest` pueue group (one
task at a time). A run already inside a queued task holds the slot already:
`sinnixd-queue-run` exports ``SINNIXD_JOB_ID``, and
``POLYLOGUE_PYTEST_SLOT=held`` is the explicit escape for a hermetic test of
this mechanism. Anything else queues and waits.

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
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final

from devtools.cloud_sentinels import cloud_sentinel_declined

__all__ = [
    "BASETEMP_ROOT_ENV",
    "INHERITED_ENVIRONMENT_KEYS",
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
QUEUE_TASK_ENV: Final = "SINNIXD_JOB_ID"

#: Explicit escape, for the hermetic test of this mechanism.
SLOT_ESCAPE_ENV: Final = "POLYLOGUE_PYTEST_SLOT"
SLOT_HELD: Final = "held"

#: The host group whose parallelism is one.
PYTEST_GROUP: Final = "pytest"

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
    return argv, contained, scratch


def holds_pytest_slot(env: Mapping[str, str]) -> bool:
    """Whether this process is already inside the host's pytest slot."""
    return bool(env.get(QUEUE_TASK_ENV)) or env.get(SLOT_ESCAPE_ENV) == SLOT_HELD


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


def _write_launch(path: Path, *, argv: Sequence[str], cwd: str, env: Mapping[str, str], log_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "argv": list(argv),
        "working_directory": cwd,
        "environment": dict(env),
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
    _write_launch(launch_path, argv=command, cwd=cwd, env=env, log_path=log_path)
    adder = adder_environment(env)
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
            command[0],
            "-m",
            "devtools.pytest_slot",
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
    _pueue(["wait", task_id], env=adder)
    status = _pueue(["status", "--json"], env=adder)
    returncode = _task_result(status.stdout, task_id)
    sys.stderr.write(f"  pytest slot released; output: {log_path}\n")
    sys.stderr.flush()
    return SlotOutcome(returncode=returncode, slot=f"pueue task {task_id}", log_path=log_path)


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

    Inside a queued task (or under the explicit escape) the slot is already
    held and the command runs here, streaming as before. Otherwise it is queued
    in the host's single-slot ``pytest`` group and its output is captured.
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
    with open(log_path, "wb") as log:
        try:
            completed = subprocess.run(
                list(launch["argv"]),
                cwd=launch["working_directory"],
                env=environment,
                stdout=log,
                stderr=log,
            )
        except OSError as exc:
            log.write(f"devtools.pytest_slot: could not start pytest: {exc}\n".encode())
            return 125
    return completed.returncode


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())

"""Bind managed execution to a private readonly copy of declared source bytes.

Filesystem notifications cannot prove execution stability (mmap and aliases can
write without a watched event). The child instead sees independent copied
inodes, mounted readonly at the original checkout path and every copy alias.
Only declared runtime bindings remain writable inside that checkout.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

from devtools.pytest_invocation import CLOSED_WORLD_COLLECTION_ARGS, effective_hypothesis_profile
from devtools.pytest_options import declared_testmon_environment
from devtools.testmon_provision import testmon_environment
from devtools.verify_runs import aggregate_pytest_statistics, git_worktree_content_sha256


class ExecutionSourceGuard:
    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        self.copy = Path(tempfile.mkdtemp(prefix="polylogue-source-", dir="/realm/tmp/work"))
        self.failure: str | None = None
        self.closed = False
        self.previous_unavailable = False
        self.graph_lock: int | None = None
        self.fds: list[int] = []
        self.infos: list[IO[bytes]] = []
        self.resources = contextlib.ExitStack()
        self.bindings: list[str] = []
        self.digest: str | None = None
        self.launch_possible = False
        self.attempts: list[dict[str, Any]] = []
        self.custody_settled = False
        self.bytecode = Path(tempfile.mkdtemp(prefix=".bytecode-", dir=self.copy))
        try:
            listed = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                cwd=root,
                capture_output=True,
                check=True,
            )
            paths = sorted(set(listed.stdout.split(b"\0")) - {b""})
            for name in paths:
                if Path(os.fsdecode(name)).parts[0] in {".git", ".venv", ".cache"}:
                    raise OSError("declared source overlaps a runtime binding")
                source, destination = self.root / os.fsdecode(name), self.copy / os.fsdecode(name)
                try:
                    mode = source.lstat().st_mode
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(mode):
                    raise OSError("declared source is not a regular file")
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Copy bytes, never links: writes through original hardlink or
                # mmap aliases cannot change the executed cohort.
                shutil.copyfile(source, destination)
                destination.chmod(stat.S_IMODE(mode))
            self.digest = git_worktree_content_sha256(self.copy, paths=paths)
            if self.digest is None:
                raise OSError("copied source could not be identified")
            # Keep these owners at their declared paths after the root overlay.
            # The fd preserves the original inode even for a worktree .git file
            # or a venv symlink whose textual path now resolves in the copy.
            for binding, writable in ((".git", False), (".venv", False), (".cache", True)):
                original = self.root / binding
                if binding == ".cache":
                    original.mkdir(exist_ok=True)
                if not original.exists():
                    continue
                destination = self.copy / binding
                if original.is_dir():
                    destination.mkdir(exist_ok=True)
                else:
                    destination.touch()
                fd = os.open(original, os.O_PATH)
                self.fds.append(fd)
                self.bindings.extend(["--bind-fd" if writable else "--ro-bind-fd", str(fd), str(self.root / binding)])
        except (OSError, subprocess.CalledProcessError) as exc:
            self.failure = str(exc)

    def command(
        self, command: Sequence[str], environment: Mapping[str, str], provenance: Mapping[str, Any] | None
    ) -> list[str]:
        from devtools.pytest_slot import PytestSlotUnavailableError

        if provenance is None or provenance.get("git_worktree_content_sha256") != self.digest:
            self.failure = "copied source differs from admitted execution content"
        declared = declared_testmon_environment(list(command))
        if declared is not None:
            profile, _source = effective_hypothesis_profile(command, environment, default="default")
            if declared != testmon_environment(self.copy, profile):
                self.failure = "executed policy differs from declared testmon environment"
        if self.failure:
            raise PytestSlotUnavailableError(self.failure)
        # The boundary owns this descriptor through child settlement.
        info = self.resources.enter_context(tempfile.TemporaryFile(dir="/realm/tmp/work"))  # noqa: SIM115
        self.infos.append(info)
        boundary = [
            "bwrap",
            "--info-fd",
            str(info.fileno()),
            "--die-with-parent",
            "--bind",
            "/",
            "/",
            "--dev-bind",
            "/dev",
            "/dev",
            "--ro-bind",
            str(self.copy),
            str(self.copy),
            "--ro-bind",
            str(self.copy),
            str(self.root),
            *self.bindings,
            # Source lookups cannot read a timestamp-valid shared pyc or
            # write a replacement into their immutable lookup directory.
            "--setenv",
            "PYTHONPYCACHEPREFIX",
            str(self.bytecode),
            "--chdir",
            str(self.root),
            "--",
            *command,
        ]

        # The supervisor is a declared source input. It is copied independently
        # and sealed before launch; Python never rereads mutable checkout code.
        supervisor = (self.copy / "devtools/execution_custody.py").read_bytes()
        executable = os.memfd_create("polylogue-execution-custody", os.MFD_ALLOW_SEALING)
        self.fds.append(executable)
        with os.fdopen(os.dup(executable), "wb") as stream:
            stream.write(supervisor)
        fcntl.fcntl(
            executable,
            fcntl.F_ADD_SEALS,
            fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL,
        )
        closure = self.resources.enter_context(tempfile.TemporaryFile(dir="/realm/tmp/work"))  # noqa: SIM115
        token = uuid.uuid4().hex
        from devtools.pytest_slot import STOP_KILL_GRACE_S, STOP_TERM_GRACE_S

        config = {
            "command": boundary,
            "child_fds": [*(int(self.bindings[index + 1]) for index in range(0, len(self.bindings), 3)), info.fileno()],
            "receipt_fd": closure.fileno(),
            "token": token,
            "term_grace_s": STOP_TERM_GRACE_S,
            "kill_grace_s": STOP_KILL_GRACE_S,
        }
        self.attempts.append({"closure": closure, "token": token, "pid": None, "pidfd": None})
        # Popen can create its child before Python assigns/records its result.
        # From this point onward an interruption cannot prove no launch; only
        # exact registered supervisor death and closure permit source removal.
        self.launch_possible = True
        return [sys.executable, "-I", "-B", f"/proc/self/fd/{executable}", json.dumps(config)]

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return (
            *self.fds,
            *(info.fileno() for info in self.infos),
            *(attempt["closure"].fileno() for attempt in self.attempts),
        )

    def launched_process(self, process: subprocess.Popen[Any]) -> None:
        self.launch_possible = True
        attempt = self.attempts[-1]
        attempt["pid"] = process.pid
        attempt["pidfd"] = os.pidfd_open(process.pid)

    def stop(self, process: subprocess.Popen[Any]) -> None:
        """Request settlement without reentering Popen's wait lock in a signal."""
        from devtools.pytest_slot import STOP_KILL_GRACE_S, STOP_TERM_GRACE_S

        attempt = next((attempt for attempt in self.attempts if attempt["pid"] == process.pid), None)
        descriptor = attempt["pidfd"] if attempt is not None else None
        borrowed = descriptor is None
        if borrowed:
            # Cancellation can arrive after Popen but before launch binding.
            # Recover only exact direct-child custody for cleanup, never the
            # missing original attempt authority. WNOWAIT leaves an exited
            # child unreaped while its birth is pinned and revalidated.
            from devtools.pytest_memory import _identity

            self.failure = "execution supervisor was not pinned"
            try:
                os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
                identity = _identity(process.pid, proc=Path("/proc"))
                if identity is None:
                    return
                descriptor = os.pidfd_open(process.pid)
                os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
                current = _identity(process.pid, proc=Path("/proc"))
                if current is None or current.start_ticks != identity.start_ticks:
                    os.close(descriptor)
                    return
            except OSError:
                if descriptor is not None:
                    os.close(descriptor)
                return
        assert descriptor is not None
        try:
            poller = select.poll()
            poller.register(descriptor, select.POLLIN)
            if not any(events & select.POLLIN for _fd, events in poller.poll(0)):
                with contextlib.suppress(ProcessLookupError):
                    signal.pidfd_send_signal(descriptor, signal.SIGTERM)
                # The supervisor has both existing descendant cleanup graces; one
                # dispatch second lets it publish closure after the final reap.
                dead = any(
                    events & select.POLLIN
                    for _fd, events in poller.poll(int((STOP_TERM_GRACE_S + STOP_KILL_GRACE_S + 1) * 1000))
                )
                if not dead:
                    self.failure = "execution supervisor settlement did not complete"
                    with contextlib.suppress(ProcessLookupError):
                        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                    poller.poll(int(STOP_KILL_GRACE_S * 1000))
            # This may run inside process.wait()'s signal handler. Kernel wait is
            # exact and nonblocking; Popen's Python lock must not delay settlement.
            with contextlib.suppress(ChildProcessError):
                pid, status = os.waitpid(process.pid, os.WNOHANG)
                if pid == process.pid:
                    process.returncode = os.waitstatus_to_exitcode(status)
        finally:
            if borrowed:
                os.close(descriptor)

    def finish(self) -> dict[str, Any]:
        if self.closed:
            return {"status": "unavailable", "observer": "readonly_snapshot", "reason": "boundary already closed"}
        self.closed = True
        for info in self.infos:
            try:
                info.seek(0)
                setup = json.load(info)
                if not isinstance(setup, dict) or not isinstance(setup.get("child-pid"), int):
                    self.failure = "readonly boundary setup was not proved"
            except (OSError, ValueError):
                self.failure = "readonly boundary setup was not proved"
            finally:
                info.close()
        closures = []
        for attempt in self.attempts:
            closure, descriptor = attempt["closure"], attempt["pidfd"]
            try:
                closure.seek(0)
                proof = json.load(closure)
                poller = select.poll()
                if descriptor is None:
                    raise ValueError("supervisor birth was not pinned")
                poller.register(descriptor, select.POLLIN)
                dead = any(events & select.POLLIN for _fd, events in poller.poll(0))
                if (
                    not isinstance(proof, dict)
                    or proof.get("token") != attempt["token"]
                    or proof.get("supervisor_pid") != attempt["pid"]
                    or proof.get("closed") is not True
                    or proof.get("proof") != "waitpid_echild"
                    or not isinstance(proof.get("main_exit"), int)
                    or proof.get("failure") is not None
                    or not dead
                ):
                    raise ValueError("kernel child closure was not proved")
                closures.append(
                    {
                        "supervisor_pid": proof["supervisor_pid"],
                        "proof": proof["proof"],
                        "main_exit": proof["main_exit"],
                    }
                )
            except (OSError, ValueError) as exc:
                self.failure = f"execution custody unavailable: {exc}"
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                closure.close()
        self.custody_settled = bool(self.attempts) and len(closures) == len(self.attempts)
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError as exc:
                self.failure = f"boundary descriptor cleanup failed: {exc}"
        self.fds.clear()
        self.resources.close()
        if not self.launch_possible or self.custody_settled:
            try:
                shutil.rmtree(self.copy)
            except OSError as exc:
                self.failure = f"boundary source cleanup failed: {exc}"
        return {
            "status": "unavailable" if self.failure or not self.launch_possible else "stable",
            "observer": "readonly_snapshot",
            "reason": self.failure or (None if self.launch_possible else "execution never launched"),
            "git_worktree_content_sha256": self.digest,
            "custody_settled": self.custody_settled,
            "attempt_closures": closures,
        }


def start_execution(root: Path, environment: Mapping[str, str]) -> ExecutionSourceGuard | None:
    if environment.get("POLYLOGUE_FOCUSED_WORKTREE_PROVENANCE") != "1":
        return None
    guard = ExecutionSourceGuard(root)
    if environment.get("TESTMON_DATAFILE"):
        marker = Path(environment["TESTMON_DATAFILE"] + ".authority-unavailable")
        try:
            guard.graph_lock = os.open(str(marker) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(guard.graph_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            guard.previous_unavailable = marker.exists()
            marker.write_text("execution source authority has not completed", encoding="utf-8")
        except OSError:
            guard.finish()
            if guard.graph_lock is not None:
                os.close(guard.graph_lock)
            from devtools.pytest_slot import PytestSlotUnavailableError

            raise PytestSlotUnavailableError("graph execution authority could not be acquired") from None
    return guard


def complete_execution(command: Sequence[str], environment: Mapping[str, str], returncode: int) -> bool:
    """An actual terminal complete run, rather than the caller's noselect intent."""
    if (
        environment.get("POLYLOGUE_TESTMON_COMPLETE") != "1"
        or "--testmon-noselect" not in command
        or "--collect-only" in command
    ):
        return False
    corpus = CLOSED_WORLD_COLLECTION_ARGS
    if not any(tuple(command[index : index + len(corpus)]) == corpus for index in range(len(command))):
        return False
    selection = environment.get("POLYLOGUE_PYTEST_SELECTION_PATH")
    if not selection:
        return False
    try:
        step_dir = Path(selection).parent
        run_id = environment.get("POLYLOGUE_VERIFY_RUN_ID")
        if not run_id:
            return False
        for name in ("selection.json", "summary.json"):
            artifact = json.loads((step_dir / name).read_text())
            if not isinstance(artifact, dict) or artifact.get("run_id") != run_id:
                return False
        evidence = aggregate_pytest_statistics(
            Path(selection).parent, command=command, step_result={"exit": returncode}
        )
    except (OSError, ValueError):
        return False
    return (
        bool(evidence["ordinary_eligible"])
        and evidence["deselected_count"] == 0
        and evidence["selected_count"] == evidence["terminal_count"]
    )


def finish_execution(
    guard: ExecutionSourceGuard | None, environment: Mapping[str, str], *, complete: bool = False
) -> dict[str, Any] | None:
    if guard is None:
        return None
    evidence = guard.finish()
    try:
        if environment.get("TESTMON_DATAFILE"):
            marker = Path(environment["TESTMON_DATAFILE"] + ".authority-unavailable")
            if evidence["status"] != "stable":
                marker.write_text(str(evidence["reason"]), encoding="utf-8")
            elif not guard.previous_unavailable or complete:
                marker.unlink(missing_ok=True)
    except OSError as exc:
        evidence = {
            "status": "unavailable",
            "observer": "readonly_snapshot",
            "reason": f"authority publication failed: {exc}",
        }
    finally:
        if guard.graph_lock is not None:
            os.close(guard.graph_lock)
            guard.graph_lock = None
    return evidence

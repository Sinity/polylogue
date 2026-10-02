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
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

from devtools.pytest_invocation import CLOSED_WORLD_COLLECTION_ARGS, effective_hypothesis_profile
from devtools.pytest_rerun import testmon_rerun_environment
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
        self.launched = False
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
        declared = testmon_rerun_environment(list(command))
        if declared is not None:
            profile, _source = effective_hypothesis_profile(command, environment, default="default")
            if declared != testmon_environment(self.copy, profile):
                self.failure = "executed policy differs from declared testmon environment"
        if self.failure:
            raise PytestSlotUnavailableError(self.failure)
        # The boundary owns this descriptor through child settlement.
        info = self.resources.enter_context(tempfile.TemporaryFile(dir="/realm/tmp/work"))  # noqa: SIM115
        self.infos.append(info)
        return [
            "bwrap",
            "--info-fd",
            str(info.fileno()),
            "--die-with-parent",
            "--unshare-pid",
            "--proc",
            "/proc",
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

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return (*self.fds, *(info.fileno() for info in self.infos))

    def finish(self) -> dict[str, Any]:
        if self.closed:
            return {"status": "unavailable", "observer": "readonly_snapshot", "reason": "boundary already closed"}
        self.closed = True
        for info in self.infos:
            try:
                info.seek(0)
                setup = json.load(info)
                if (
                    not isinstance(setup, dict)
                    or not isinstance(setup.get("child-pid"), int)
                    or not isinstance(setup.get("pid-namespace"), int)
                ):
                    self.failure = "readonly boundary setup was not proved"
                else:
                    # bwrap reports the namespace init. Its death settles all
                    # namespace descendants in the kernel, including setsid
                    # children; a live same namespace cannot publish authority.
                    try:
                        namespace = Path(f"/proc/{setup['child-pid']}/ns/pid").stat().st_ino
                    except FileNotFoundError:
                        namespace = None
                    if namespace == setup["pid-namespace"]:
                        self.failure = "execution PID namespace remains alive"
            except (OSError, ValueError):
                self.failure = "readonly boundary setup was not proved"
            finally:
                info.close()
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError as exc:
                self.failure = f"boundary descriptor cleanup failed: {exc}"
        self.fds.clear()
        self.resources.close()
        try:
            shutil.rmtree(self.copy)
        except OSError as exc:
            self.failure = f"boundary source cleanup failed: {exc}"
        return {
            "status": "unavailable" if self.failure or not self.launched else "stable",
            "observer": "readonly_snapshot",
            "reason": self.failure or (None if self.launched else "execution never launched"),
            "git_worktree_content_sha256": self.digest,
            "pid_namespace_settled": self.launched and bool(self.infos) and self.failure is None,
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

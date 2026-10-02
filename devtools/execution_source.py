"""Observe mutations of declared checkout sources across managed execution.

Linux inotify queues mutations even when bytes are restored before the next
sample. Overflow, watch removal and unavailable observation refuse authority.
Generated ignored files are outside the declared source contract.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import struct
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ExecutionSourceGuard:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.fd = -1
        self.directories: dict[int, Path] = {}
        self.paths: set[Path] = set()
        self.failure: str | None = None
        self.closed = False
        self.previous_unavailable = False
        self.graph_lock: int | None = None
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            self.fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
            if self.fd < 0:
                raise OSError(ctypes.get_errno(), "inotify initialization failed")
            listed = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                cwd=root,
                capture_output=True,
                check=True,
            )
            self.paths = {Path(os.fsdecode(name)) for name in listed.stdout.split(b"\0") if name}
            directories = {Path(".")}
            for path in self.paths:
                if (root / path).is_symlink():
                    raise OSError("declared symlink source cannot be observed")
                directories.update(path.parents)
            # Parents first. A new/replaced directory after registration is
            # itself a mutation; no recursive late-registration race is accepted.
            for directory in sorted(directories, key=lambda path: len(path.parts)):
                target = root / directory
                if not target.is_dir():
                    continue
                watch = libc.inotify_add_watch(self.fd, os.fsencode(target), 0x00000FC6)
                if watch < 0:
                    raise OSError(ctypes.get_errno(), "source watch registration failed")
                self.directories[watch] = directory
        except (AttributeError, OSError, subprocess.CalledProcessError) as exc:
            self.failure = str(exc)

    def finish(self) -> dict[str, Any]:
        if self.closed:
            return {"status": "unavailable", "observer": "inotify", "reason": "observer already closed"}
        self.closed = True
        try:
            while self.fd >= 0:
                try:
                    events = os.read(self.fd, 65536)
                except BlockingIOError:
                    break
                if not events:
                    self.failure = "source event stream closed"
                    break
                offset = 0
                while offset < len(events):
                    watch, mask, _cookie, length = struct.unpack_from("iIII", events, offset)
                    name = os.fsdecode(events[offset + 16 : offset + 16 + length].split(b"\0", 1)[0])
                    offset += 16 + length
                    if mask & (0x00004000 | 0x00008000 | 0x00000800 | 0x00000400):
                        self.failure = "source event coverage lost"
                        continue
                    path = self.directories.get(watch, Path(".")) / name
                    if path in self.paths:
                        self.failure = "declared source mutated during execution"
                    else:
                        ignored = subprocess.run(
                            [
                                "git",
                                "check-ignore",
                                "--no-index",
                                "--quiet",
                                "--",
                                path.as_posix() + ("/" if mask & 0x40000000 else ""),
                            ],
                            cwd=self.root,
                            check=False,
                        )
                        if ignored.returncode != 0:
                            self.failure = f"declared source membership changed during execution: {path}"
        except (OSError, ValueError, struct.error) as exc:
            self.failure = str(exc)
        finally:
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1
        return {"status": "unavailable" if self.failure else "stable", "observer": "inotify", "reason": self.failure}


def start_execution(root: Path, environment: Mapping[str, str]) -> ExecutionSourceGuard | None:
    if environment.get("POLYLOGUE_FOCUSED_WORKTREE_PROVENANCE") != "1":
        return None
    guard = ExecutionSourceGuard(root)
    guard.previous_unavailable = False
    if environment.get("TESTMON_DATAFILE"):
        marker = Path(environment["TESTMON_DATAFILE"] + ".authority-unavailable")
        try:
            guard.graph_lock = os.open(str(marker) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(guard.graph_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            guard.previous_unavailable = marker.exists()
            marker.write_text("execution source observation has not completed", encoding="utf-8")
        except OSError:
            guard.finish()
            if guard.graph_lock is not None:
                os.close(guard.graph_lock)
            from devtools.pytest_slot import PytestSlotUnavailableError

            raise PytestSlotUnavailableError("graph execution authority could not be acquired") from None
    return guard


def finish_execution(guard: ExecutionSourceGuard | None, environment: Mapping[str, str]) -> dict[str, Any] | None:
    if guard is None:
        return None
    evidence = guard.finish()
    try:
        if environment.get("TESTMON_DATAFILE"):
            marker = Path(environment["TESTMON_DATAFILE"] + ".authority-unavailable")
            if evidence["status"] != "stable":
                marker.write_text(str(evidence["reason"]), encoding="utf-8")
            elif not guard.previous_unavailable or environment.get("POLYLOGUE_TESTMON_COMPLETE") == "1":
                marker.unlink(missing_ok=True)
    except OSError as exc:
        evidence = {"status": "unavailable", "observer": "inotify", "reason": f"authority publication failed: {exc}"}
    finally:
        if guard.graph_lock is not None:
            os.close(guard.graph_lock)
            guard.graph_lock = None
    return evidence

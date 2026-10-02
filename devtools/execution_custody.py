"""Standalone kernel descendant owner, executed only from sealed source bytes."""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def supervise(config: dict[str, Any]) -> int:
    """Reap the sole execution tree, including orphaned detached descendants."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "subreaper setup failed")
    if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
        raise OSError(ctypes.get_errno(), "private closure custody failed")
    receipt = int(config["receipt_fd"])
    os.set_inheritable(receipt, False)
    interrupted = False

    def request_stop(_number: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, request_stop)
    child = subprocess.Popen(config["command"], pass_fds=tuple(config["child_fds"]))
    main_exit: int | None = None
    closed = False
    failure: str | None = None
    stage = 0
    deadline: float | None = None
    pinned: dict[int, int] = {}
    try:
        while True:
            # No other code creates children or reaps them. A child PID cannot
            # be reused before our wait; pidfds keep signals bound after wait.
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    closed = True
                    break
                if pid == 0:
                    break
                if pid == child.pid:
                    main_exit = os.waitstatus_to_exitcode(status)
                    child.returncode = main_exit
                descriptor = pinned.pop(pid, None)
                if descriptor is not None:
                    os.close(descriptor)
            if closed:
                break
            if main_exit is not None or interrupted:
                if deadline is None:
                    deadline = time.monotonic() + float(config["term_grace_s"])
                if time.monotonic() >= deadline:
                    if stage:
                        failure = "owned descendants did not terminate"
                        break
                    stage = 1
                    deadline = time.monotonic() + float(config["kill_grace_s"])
                # This kernel list is our direct/adopted children, never the
                # host's same-UID population. Repeated adoption catches forks
                # made in a TERM handler before the parent physically dies.
                children = Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()
                for item in children:
                    pid = int(item)
                    if pid not in pinned:
                        try:
                            pinned[pid] = os.pidfd_open(pid)
                        except ProcessLookupError:
                            continue
                    with contextlib.suppress(ProcessLookupError):
                        signal.pidfd_send_signal(pinned[pid], signal.SIGKILL if stage else signal.SIGTERM)
            time.sleep(0.01)
    except OSError as exc:
        failure = f"kernel child custody unavailable: {exc}"
    finally:
        for descriptor in pinned.values():
            os.close(descriptor)
    result = {
        "token": config["token"],
        "supervisor_pid": os.getpid(),
        "child_pid": child.pid,
        "closed": closed,
        "proof": "waitpid_echild" if closed else None,
        "main_exit": main_exit,
        "failure": failure,
    }
    os.write(receipt, json.dumps(result).encode())
    os.close(receipt)
    if not closed or main_exit is None:
        return 125
    return main_exit


if __name__ == "__main__":
    code = supervise(json.loads(sys.argv[1]))
    if code < 0:
        # Preserve Popen's signal outcome as well as ordinary exit statuses.
        number = -code
        if number not in {signal.SIGKILL, signal.SIGSTOP}:
            signal.signal(number, signal.SIG_DFL)
        os.kill(os.getpid(), number)
    raise SystemExit(code)

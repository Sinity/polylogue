"""Per-process measurements for subprocess benchmarks."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from tempfile import TemporaryFile


def run_with_peak_rss(
    argv: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: str | os.PathLike[str] | None = None
) -> tuple[subprocess.CompletedProcess[bytes], int]:
    """Return this child's output and peak RSS in KiB, not a children-wide delta.

    Temporary files avoid pipe backpressure while wait4 owns reaping. No other
    wait or poll can consume the child's individual usage record first.
    """
    with TemporaryFile() as stdout, TemporaryFile() as stderr:
        with subprocess.Popen(argv, env=env, cwd=cwd, stdout=stdout, stderr=stderr) as child:
            try:
                _, status, usage = os.wait4(child.pid, 0)
                child.returncode = os.waitstatus_to_exitcode(status)
            except BaseException:
                child.kill()
                child.wait()
                raise
            stdout.seek(0)
            stderr.seek(0)
            result = subprocess.CompletedProcess(argv, child.returncode, stdout.read(), stderr.read())
        peak_rss = int(usage.ru_maxrss)
        return result, peak_rss // 1024 if sys.platform == "darwin" else peak_rss

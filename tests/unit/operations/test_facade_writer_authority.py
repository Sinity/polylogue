"""Embedded-facade writers refuse an archive a resident daemon owns.

Each durable writer behind the embedded Python facade must check archive
writer ownership before it opens or creates a tier file. A writer that skips
the check would open ``user.db``/``ops.db``/``index.db`` beside a live
``polylogued run`` from another process, which is exactly what the single
writer contract forbids.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.maintenance.offline_guard import ArchiveWriterOwnershipError
from polylogue.operations import facade_writers


@pytest.fixture
def resident_daemon(tmp_path: Path) -> Iterator[Callable[[Path], int]]:
    """Hold the daemon's exclusive pidfile lock from a separate process."""
    processes: list[subprocess.Popen[str]] = []
    script = tmp_path / "hold_pidfile.py"
    script.write_text(
        "import fcntl, os, sys, time\n"
        "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)\n"
        "fcntl.flock(fd, fcntl.LOCK_EX)\n"
        "os.write(fd, str(os.getpid()).encode())\n"
        "os.fsync(fd)\n"
        "sys.stdout.write('ready\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(300)\n"
    )

    def start(pidfile: Path) -> int:
        process = subprocess.Popen([sys.executable, str(script), str(pidfile)], stdout=subprocess.PIPE, text=True)
        processes.append(process)
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        return process.pid

    try:
        yield start
    finally:
        for process in processes:
            process.kill()
            process.wait(timeout=30)
            if process.stdout is not None:
                process.stdout.close()


_WRITERS: dict[str, Callable[[Config], object]] = {
    "manual_continuation": lambda config: facade_writers.record_manual_continuation_product(
        config, "claude-code:child", "claude-code:parent"
    ),
    "context_ledger": lambda config: facade_writers.record_context_ledger_product(config, object(), observed_at_ms=1),
    "comparative_judgment": lambda config: facade_writers._archive_record_comparative_judgment(
        config, object(), author_kind="human"
    ),
}


@pytest.mark.parametrize("writer", sorted(_WRITERS))
def test_facade_writer_refuses_a_daemon_owned_archive_before_touching_a_tier(
    tmp_path: Path,
    resident_daemon: Callable[[Path], int],
    writer: str,
) -> None:
    """Anti-vacuity: drop the writer's ``require_archive_write_authority`` call
    and it opens or creates a tier file instead, so the call either raises a
    SQLite/ValueError rather than the ownership refusal, or leaves a ``*.db``
    file in the archive root.
    """
    root = tmp_path / "archive"
    root.mkdir()
    resident_pid = resident_daemon(root / "daemon.pid")
    config = Config(archive_root=root, render_root=root, sources=[])

    with pytest.raises(ArchiveWriterOwnershipError) as caught:
        _WRITERS[writer](config)

    assert f"PID {resident_pid}" in str(caught.value)
    assert sorted(path.name for path in root.glob("*.db")) == []

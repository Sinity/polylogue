"""Independent-process proof of the archive's physical write exclusion."""

from __future__ import annotations

from pathlib import Path

from polylogue.storage.sqlite.write_lease import ARCHIVE_WRITE_CUSTODY_LOCK_NAME


def archive_custody_available(root: Path) -> bool:
    import subprocess
    import sys

    probe = (
        "import fcntl, os, sys\n"
        "fd = os.open(sys.argv[1], os.O_RDWR)\n"
        "try:\n"
        "    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "except BlockingIOError:\n"
        "    sys.exit(1)\n"
        "finally:\n"
        "    os.close(fd)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe, str(root / ARCHIVE_WRITE_CUSTODY_LOCK_NAME)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stderr == ""
    assert result.returncode in (0, 1)
    return result.returncode == 0

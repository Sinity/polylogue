"""Guards for offline maintenance that must not race a live daemon."""

from __future__ import annotations

import os
from pathlib import Path

from polylogue.config import Config


def resident_daemon_pid(archive_root: Path) -> int | None:
    """Return a live polylogued PID owning ``archive_root``, if one is present.

    The root-shaped probe, so a caller that has only resolved an archive root
    -- the CLI writer-ownership boundary does exactly that, before any config
    object exists -- asks the same question as the config-shaped callers
    rather than reimplementing the pidfile contract.
    """
    pidfile = archive_root / "daemon.pid"
    try:
        pid = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return pid if b"polylogued" in cmdline else None


def running_daemon_pid(config: Config) -> int | None:
    """Return a live polylogued PID for this archive, if one is present."""
    return resident_daemon_pid(config.archive_root)


def offline_writer_block_reason(config: Config) -> str | None:
    """Return the concrete writer that makes a strictly offline operation unsafe."""
    from polylogue.daemon.write_coordinator import daemon_write_lease_active

    if daemon_write_lease_active():
        return "a daemon writer lease is active"
    daemon_pid = running_daemon_pid(config)
    if daemon_pid is not None:
        return f"live pidfile PID {daemon_pid} is running"
    return None


def offline_maintenance_block_reason(
    config: Config,
    *,
    active: bool,
    dry_run: bool,
) -> str | None:
    """Return a refusal reason when offline maintenance would race the daemon."""
    if dry_run or not active:
        return None
    # A daemon-owned writer is already serialized against every other archive
    # mutation.  Treat it as the online equivalent of the offline exclusion
    # boundary instead of rejecting the daemon's own convergence work.
    from polylogue.daemon.write_coordinator import daemon_write_lease_active

    if daemon_write_lease_active():
        return None
    daemon_pid = running_daemon_pid(config)
    if daemon_pid is None:
        return None
    return (
        f"Refusing offline maintenance while polylogued PID {daemon_pid} is running. "
        "Stop polylogued to run this operation offline, or let daemon convergence drain live work."
    )


__all__ = [
    "offline_maintenance_block_reason",
    "offline_writer_block_reason",
    "resident_daemon_pid",
    "running_daemon_pid",
]

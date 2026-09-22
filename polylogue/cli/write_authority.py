"""The ordinary CLI process's archive writer ownership.

:mod:`polylogue.storage.sqlite.write_lease` and
:mod:`polylogue.storage.sqlite.write_guard` make every writable archive-tier
open assert ownership -- but only *where enforcement is armed*. Until this
module existed, enforcement was armed in exactly two processes: ``polylogued
run`` (:mod:`polylogue.daemon.cli`) and the MCP stdio bridge holding a write
or maintenance capability (:mod:`polylogue.mcp.server`). The console scripts
``polylogue``/``plg``/``plog`` armed neither, so ``write_lease_enforced()``
was ``False`` for the whole invocation and every ``require_write_lease`` call
reached on that route returned ``None``. An ordinary
``polylogue ops maintenance archive-init --yes`` therefore created and wrote
the six durable tier files beside a live daemon with no ownership check at
all -- the same unserialized-writer shape that locked the daemon out of its
own catch-up chunk (polylogue-8qm4k).

The boundary is armed on the fact that decides whether this process may own
the archive at all: **is a resident ``polylogued`` running for this root?**

* No resident daemon. The CLI *is* the archive's single writer, which is the
  standing rationale in :mod:`~polylogue.storage.sqlite.write_lease` for
  leaving one-shot writers unarmed, and the ownership the declared offline
  authorities (archive initialization, durable tier migration, embedding
  backfill and preservation) already rely on. There is no second writer to
  be serialized against, so nothing is armed and nothing changes.
* A resident daemon. The CLI is **not** the single writer, and the premise
  that excused leaving it unarmed is false. Enforcement and the
  connection-level guard are armed for the rest of the invocation, so a
  writable tier open raises before the connection exists instead of
  contending through the busy timeout, and the refusal is re-raised naming
  the resident writer that holds the archive.

Reads are untouched in both cases: a read-only open never consults the lease,
and the CLI's read and mutation routes already reach a resident daemon
through the operation kernel rather than the archive files.

This is deliberately not a lease taken over the whole invocation. The CLI
runs writes from ``asyncio`` tasks and worker threads, and a lease minted at
the entry point is bound to the thread and task that minted it -- an
invocation-wide lease turns "no ownership" into "ownership held by the wrong
task" and refuses exactly the offline authorities that are entitled to write.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

__all__ = [
    "ArchiveWriterOwnershipError",
    "cli_archive_writer_ownership",
    "resident_archive_writer",
]


class ArchiveWriterOwnershipError(RuntimeError):
    """A CLI write was refused because a resident daemon owns the archive."""

    code = "archive_writer_ownership_unavailable"


def resident_archive_writer() -> tuple[Path, str] | None:
    """Name the archive root and the resident daemon that owns it, if any."""
    from polylogue.maintenance.offline_guard import resident_daemon_pid

    try:
        from polylogue.paths import archive_root

        root = Path(archive_root())
    except Exception:
        # An unresolvable archive root is a configuration failure the invoked
        # command reports far better than this boundary can, and a process
        # that cannot name an archive cannot be racing a daemon over one.
        return None
    pid = resident_daemon_pid(root)
    if pid is None:
        return None
    return root, f"polylogued PID {pid} is running for this archive"


@contextmanager
def cli_archive_writer_ownership() -> Iterator[None]:
    """Hold the single-writer boundary for one CLI invocation.

    Entered once per invocation from the root Click callback, which is the
    only point every CLI route passes: the ``polylogue``/``plg``/``plog``
    console scripts, ``python -m polylogue`` and an embedded caller driving
    ``polylogue.cli.cli`` directly all reach it.
    """
    resident = resident_archive_writer()
    if resident is None:
        yield
        return

    root, reason = resident
    from polylogue.core.write_lease import (
        UnleasedWriteError,
        arm_write_lease_enforcement,
        install_archive_write_guard,
    )

    with ExitStack() as stack:
        stack.enter_context(arm_write_lease_enforcement(process_wide=True))
        # Arming alone only covers the declared write-mode factories; the
        # guard makes the boundary total at ``sqlite3.connect`` itself.
        stack.enter_context(install_archive_write_guard())
        try:
            yield
        except UnleasedWriteError as exc:
            raise ArchiveWriterOwnershipError(
                f"this CLI process may not write {root}: {reason}. "
                "Route the mutation through the resident daemon, or stop it to run "
                f"this operation as the archive's exclusive offline owner ({exc})"
            ) from exc

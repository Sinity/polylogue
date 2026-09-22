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

* A resident daemon at entry. The CLI is **not** the single writer, and the
  premise that excused leaving it unarmed is false. Enforcement and the
  connection-level guard are armed for the rest of the invocation, so a
  writable tier open raises before the connection exists instead of
  contending through the busy timeout, and the refusal is re-raised naming
  the resident writer that holds the archive.
* No resident daemon at entry. The CLI *is* the archive's single writer, which
  is the standing rationale in :mod:`~polylogue.storage.sqlite.write_lease`
  for leaving one-shot writers unarmed, and the ownership the declared offline
  authorities (archive initialization, durable tier migration, embedding
  backfill and preservation) already rely on. Nothing is armed -- but the
  question is **re-asked at every writable archive-tier open**, because it is
  a claim about volatile state that entry cannot settle for the whole command.
* The platform cannot answer. The boundary **refuses loudly**. An unguarded
  durable write beside a live daemon is the outcome this module exists to
  prevent, so an unprovable owner is never treated as an absent one.

Reads are untouched in every case: a read-only open never consults the lease
and is never intercepted, and the CLI's read and mutation routes already reach
a resident daemon through the operation kernel rather than the archive files.

This is deliberately not a lease taken over the whole invocation. The CLI
runs writes from ``asyncio`` tasks and worker threads, and a lease minted at
the entry point is bound to the thread and task that minted it -- an
invocation-wide lease turns "no ownership" into "ownership held by the wrong
task" and refuses exactly the offline authorities that are entitled to write.

**Why the offline case re-asks rather than trusting entry.** A single probe in
the root callback is a time-of-check/time-of-use window as wide as the command:
``polylogue ops embed backfill`` can sit at its confirmation prompt for
minutes, and a ``polylogued run`` started in that time was never noticed, so
the backfill opened writable embedding/index/ops tiers beside the live daemon.
Residency is volatile state; only the archive root is configuration. So the
root is resolved once and residency is re-asked at each open.

The offline interception itself is
:func:`~polylogue.maintenance.offline_guard.refuse_writable_tier_opens`, which
lives beside the residency probe rather than here: it needs storage's own
definition of "a writable archive-tier open", and a fresh ``cli -> storage``
import edge is what the surface layering ratchet forbids. It is deliberately
not a hook parameter on
:func:`~polylogue.storage.sqlite.write_guard.install_archive_write_guard`
either -- that module is inside the derived-schema identity closure
(``devtools schema closure``), so giving it a new parameter would move the
identity hash of every derived tier and force a fleet-wide reconvergence for a
CLI-side policy.

Residual, not closed here: a writable connection opened while no daemon was
resident stays open across a daemon start. Closing that needs an exclusion
token held for the *connection's* life, which is the invocation-wide lease
this module cannot take for the reason above.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

__all__ = [
    "ArchiveWriterOwnershipError",
    "ArchiveWriterOwnershipUndecidableError",
    "cli_archive_writer_ownership",
    "resident_archive_writer",
]


class ArchiveWriterOwnershipError(RuntimeError):
    """A CLI write was refused because a resident daemon owns the archive."""

    code = "archive_writer_ownership_unavailable"


class ArchiveWriterOwnershipUndecidableError(ArchiveWriterOwnershipError):
    """A CLI write was refused because ownership could not be proven at all.

    The boundary fails **closed**. A platform that cannot answer "does a live
    daemon own this archive?" gets a loud refusal naming the reason, never a
    silently disarmed boundary. Before this existed the probe read
    ``/proc/<pid>/cmdline`` and swallowed ``OSError``, so on macOS -- a
    supported install target (``docs/installation.md``) -- every pid answered
    "no daemon" and the boundary armed nothing on every single invocation.
    """

    code = "archive_writer_ownership_undecidable"


def _archive_root() -> Path | None:
    """Resolve the archive root once for an invocation, or ``None``.

    An unresolvable archive root is a configuration failure the invoked
    command reports far better than this boundary can, and a process that
    cannot name an archive cannot be racing a daemon over one.
    """
    try:
        from polylogue.paths import archive_root

        return Path(archive_root())
    except Exception:
        return None


def resident_archive_writer(root: Path | None = None) -> tuple[Path, str] | None:
    """Name the archive root and the resident daemon that owns it, if any.

    Raises :class:`ArchiveWriterOwnershipUndecidableError` when the platform cannot
    answer the question; returning ``None`` there would disarm the boundary.
    """
    from polylogue.maintenance.offline_guard import DaemonResidencyUndecidableError, resident_daemon_pid

    resolved = _archive_root() if root is None else root
    if resolved is None:
        return None
    try:
        pid = resident_daemon_pid(resolved)
    except DaemonResidencyUndecidableError as exc:
        raise ArchiveWriterOwnershipUndecidableError(
            f"this CLI process cannot prove whether a resident daemon owns {resolved}: {exc}. "
            "Refusing rather than writing durable tiers beside a writer this platform cannot see"
        ) from exc
    if pid is None:
        return None
    return resolved, f"polylogued PID {pid} is running for this archive"


@contextmanager
def cli_archive_writer_ownership() -> Iterator[None]:
    """Hold the single-writer boundary for one CLI invocation.

    Entered once per invocation from the root Click callback, which is the
    only point every CLI route passes: the ``polylogue``/``plg``/``plog``
    console scripts, ``python -m polylogue`` and an embedded caller driving
    ``polylogue.cli.cli`` directly all reach it.
    """
    from polylogue.maintenance.offline_guard import refuse_writable_tier_opens

    root = _archive_root()
    if root is None:
        yield
        return

    resident = resident_archive_writer(root)
    if resident is None:
        # Not armed -- there is no second writer to serialize against right
        # now. But "right now" is all entry can establish, so every writable
        # archive-tier open re-asks before it happens.
        def refuse_a_later_arrival(path: Path) -> None:
            arrived = resident_archive_writer(root)
            if arrived is None:
                return
            owned_root, reason = arrived
            raise ArchiveWriterOwnershipError(
                f"this CLI process may not write {path}: {reason}. The daemon started after this "
                f"command did, so {owned_root} is no longer this process's to write. Route the "
                "mutation through the resident daemon, or stop it and run the command again as "
                "the archive's exclusive offline owner"
            )

        with refuse_writable_tier_opens(refuse_a_later_arrival):
            yield
        return

    owned_root, reason = resident
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
                f"this CLI process may not write {owned_root}: {reason}. "
                "Route the mutation through the resident daemon, or stop it to run "
                f"this operation as the archive's exclusive offline owner ({exc})"
            ) from exc

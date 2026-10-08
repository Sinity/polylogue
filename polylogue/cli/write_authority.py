"""The ordinary CLI process's archive writer ownership.

The configured archive is written only by ``polylogued``. The CLI owns syntax,
confirmation and rendering; every mutation lowers onto a declared daemon
operation. This module enforces that for every writable archive-tier open the
CLI process makes, decided on one fact: **is a resident ``polylogued`` running
for the configured root at entry?**

* A resident daemon at entry. The same per-open ownership check refuses
  configured-archive writes and names the resident writer. Separately owned
  scratch archives remain eligible.
* No resident daemon at entry. Every writable archive-tier open is
  intercepted (:func:`~polylogue.maintenance.offline_guard.refuse_writable_tier_opens`).
  An open inside the configured archive is refused with ``daemon_required``
  -- the CLI has no offline writer for it, empty or not -- or, if a daemon
  has arrived since entry, with an ownership refusal naming it. An open
  outside the configured archive is admitted only for a scratch archive held
  by a scoped one-shot archive owner or a matching write lease plus physical
  custody. Demo seeding (``demo seed``/``receipts``/``tour``) uses the
  one-shot owner. Anything else has no owner and is refused.
* The platform cannot answer the residency question. The boundary **refuses
  loudly**; an unprovable owner is never treated as an absent one.

Reads are untouched in every case: a read-only open never consults the lease
and is never intercepted.

This is deliberately not a lease taken over the whole invocation: a lease
minted at the entry point is bound to the thread and task that minted it,
while the CLI runs work from ``asyncio`` tasks and worker threads.

The interception lives in :mod:`polylogue.maintenance.offline_guard` beside
the residency probe because it needs storage's own definition of "a writable
archive-tier open", and a fresh ``cli -> storage`` import edge is what the
surface layering ratchet forbids.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "ArchiveWriterOwnershipError",
    "ArchiveWriterOwnershipUndecidableError",
    "cli_archive_writer_ownership",
    "resident_archive_writer",
]


from polylogue.maintenance.offline_guard import (
    ArchiveWriterOwnershipError,
    ArchiveWriterOwnershipUndecidableError,
)


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
            "Refusing rather than writing durable tiers beside a writer this platform cannot see",
            archive_root=resolved,
        ) from exc
    if pid is None:
        return None
    return resolved, f"polylogued PID {pid} is running for this archive"


def _refuse_configured_archive_write(path: Path, root: Path) -> None:
    """Refuse a writable open inside the configured archive from this process.

    The configured archive is written only by ``polylogued``; the CLI has no
    offline owner for it, empty or not. A daemon that is resident by now is
    named so the operator knows which writer to route to.
    """
    from polylogue.cli.shared.helpers import DaemonRequiredError

    arrived = resident_archive_writer(root)
    if arrived is not None:
        owned_root, reason = arrived
        raise ArchiveWriterOwnershipError(
            f"this CLI process may not write {path}: {reason}. Route the mutation through the resident daemon",
            archive_root=owned_root,
            resident_writer=reason,
        )
    raise DaemonRequiredError(
        f"this CLI process may not write {path}: the configured archive {root} is written only by "
        "`polylogued run`, and the CLI has no offline writer for it. Start the daemon and repeat the command",
        archive_root=root,
    )


def _require_scratch_archive_owner(path: Path, *, configured_root: Path) -> None:
    """Admit a writable open outside the configured archive only for its owner.

    Demo seeding (``demo seed``/``receipts``/``tour``) builds a synthetic
    scratch archive under :func:`~polylogue.maintenance.offline_guard.scoped_offline_archive_writer`, which
    holds both the daemon-start exclusion and the archive's physical identity
    claim. A separate operation can also own its scratch root with a matching
    write lease and SQL custody. Any other writable tier open from the CLI has
    no owner and is refused.
    """
    from polylogue.core.write_lease import current_sql_custody, current_write_lease, require_write_lease
    from polylogue.maintenance.offline_guard import current_offline_archive_writer_root

    lease = current_write_lease()
    lease_root = lease.archive_root.resolve() if lease is not None and lease.archive_root is not None else None
    offline_root = current_offline_archive_writer_root()
    if offline_root is not None and lease_root is not None and offline_root != lease_root:
        raise ArchiveWriterOwnershipError(
            f"this CLI process has overlapping archive owners for {path}: offline owner {offline_root}, "
            f"write lease {lease_root}",
            archive_root=offline_root,
        )
    owner_root = lease_root or offline_root
    if owner_root is None or not path.resolve().is_relative_to(owner_root):
        raise ArchiveWriterOwnershipError(
            f"this CLI process may not write {path}: only a separately owned scratch archive is writable from the CLI",
            archive_root=path.parent,
        )
    if (
        configured_root == owner_root
        or configured_root.is_relative_to(owner_root)
        or owner_root.is_relative_to(configured_root)
    ):
        raise ArchiveWriterOwnershipError(
            f"this CLI process may not write {path}: its scratch archive root {owner_root} overlaps the "
            f"configured archive {configured_root}",
            archive_root=configured_root,
        )
    arrived = resident_archive_writer(owner_root)
    if arrived is not None:
        owned_root, reason = arrived
        raise ArchiveWriterOwnershipError(
            f"this CLI process may not write {path}: {reason}. Route the mutation through that daemon",
            archive_root=owned_root,
            resident_writer=reason,
        )
    if offline_root is not None and lease_root is None:
        # The scoped offline owner holds the shared daemon.pid lock and the
        # OwnedArchiveLocation claim for this exact root. CLI lease enforcement
        # stays unarmed in this branch, so no write lease is involved.
        return
    require_write_lease("CLI scratch archive writer", archive_root=lease_root)
    custody = current_sql_custody()
    if custody is None:
        raise ArchiveWriterOwnershipError(
            "this CLI writer has no current physical archive custody",
            archive_root=lease_root,
        )
    custody.assert_namespace()


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

        def refuse_unresolved_root_write(path: Path) -> None:
            raise ArchiveWriterOwnershipUndecidableError(
                f"this CLI process may not write {path}: it cannot resolve the configured archive root, "
                "so it cannot prove this is a nonconfigured scratch archive",
                archive_root=path.parent,
            )

        with refuse_writable_tier_opens(refuse_unresolved_root_write):
            yield
        return

    resident = resident_archive_writer(root)
    resolved_root = root.resolve()

    def refuse_unowned_write(path: Path) -> None:
        from polylogue.core.write_lease import coordinator_write_lease_active, require_write_lease

        # An embedded daemon can serve this CLI from another worker in the
        # same process. Its admitted coordinator retains its own authority.
        if coordinator_write_lease_active():
            require_write_lease("CLI shared-process daemon writer", archive_root=resolved_root)
            return
        if path.resolve().is_relative_to(resolved_root):
            _refuse_configured_archive_write(path, resolved_root)
        _require_scratch_archive_owner(path, configured_root=resolved_root)

    from polylogue.core.write_lease import UnleasedWriteError

    with refuse_writable_tier_opens(refuse_unowned_write):
        try:
            yield
        except UnleasedWriteError as exc:
            if resident is None:
                raise
            owned_root, reason = resident
            raise ArchiveWriterOwnershipError(
                f"this CLI process may not write {owned_root}: {reason}. "
                f"Route the mutation through the resident daemon ({exc})",
                archive_root=owned_root,
                resident_writer=reason,
            ) from exc

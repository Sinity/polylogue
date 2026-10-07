"""The ordinary CLI process's archive writer ownership.

The configured archive is written only by ``polylogued``. The CLI owns syntax,
confirmation and rendering; every mutation lowers onto a declared daemon
operation. This module enforces that for every writable archive-tier open the
CLI process makes, decided on one fact: **is a resident ``polylogued`` running
for the configured root at entry?**

* A resident daemon at entry. Write-lease enforcement and the
  connection-level guard are armed for the invocation, so an unleased
  writable open raises before the connection exists, and the refusal is
  re-raised naming the resident writer.
* No resident daemon at entry. Every writable archive-tier open is
  intercepted (:func:`~polylogue.maintenance.offline_guard.refuse_writable_tier_opens`).
  An open inside the configured archive is refused with ``daemon_required``
  -- the CLI has no offline writer for it, empty or not -- or, if a daemon
  has arrived since entry, with an ownership refusal naming it. An open
  outside the configured archive is admitted only for a scratch archive held
  under this command's own write lease and physical custody, which is how
  demo seeding (``demo seed``/``receipts``/``tour``) builds its synthetic
  root. Anything else has no owner and is refused.
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
surface layering ratchet forbids. It is not a hook parameter on
:func:`~polylogue.storage.sqlite.write_guard.install_archive_write_guard`
either: that module is inside the derived-schema identity closure.
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
    """Admit a writable open outside the configured archive only for its leased owner.

    Demo seeding (``demo seed``/``receipts``/``tour``) builds a synthetic
    scratch archive under a write lease bound to that root, with the root's
    physical custody held by :func:`~polylogue.maintenance.offline_guard.scoped_offline_archive_writer`.
    Any other writable tier open from the CLI has no owner and is refused.
    """
    from polylogue.core.write_lease import current_sql_custody, current_write_lease, require_write_lease

    lease = current_write_lease()
    lease_root = lease.archive_root.resolve() if lease is not None and lease.archive_root is not None else None
    if lease_root is None or not path.resolve().is_relative_to(lease_root):
        raise ArchiveWriterOwnershipError(
            f"this CLI process may not write {path}: only a scratch archive owned under this "
            "command's own write lease is writable from the CLI",
            archive_root=path.parent,
        )
    if (
        configured_root == lease_root
        or configured_root.is_relative_to(lease_root)
        or lease_root.is_relative_to(configured_root)
    ):
        raise ArchiveWriterOwnershipError(
            f"this CLI process may not write {path}: its scratch archive root {lease_root} overlaps the "
            f"configured archive {configured_root}",
            archive_root=configured_root,
        )
    arrived = resident_archive_writer(lease_root)
    if arrived is not None:
        owned_root, reason = arrived
        raise ArchiveWriterOwnershipError(
            f"this CLI process may not write {path}: {reason}. Route the mutation through that daemon",
            archive_root=owned_root,
            resident_writer=reason,
        )
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
    if resident is None:
        resolved_root = root.resolve()

        def refuse_unowned_write(path: Path) -> None:
            if path.resolve().is_relative_to(resolved_root):
                _refuse_configured_archive_write(path, resolved_root)
            _require_scratch_archive_owner(path, configured_root=resolved_root)

        with refuse_writable_tier_opens(refuse_unowned_write):
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
                f"Route the mutation through the resident daemon ({exc})",
                archive_root=owned_root,
                resident_writer=reason,
            ) from exc

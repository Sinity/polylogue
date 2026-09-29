"""Archive-specific writer ownership for ordinary CLI invocations.

Every writable tier open is checked against the archive that owns that file,
not merely the configured default. A resident daemon excludes an unleased
CLI writer; an offline writer holds that archive's daemon-start exclusion
until the invocation exits. Explicit scratch archives remain independent.

Storage's archive identity owner resolves promoted indexes and tier aliases.
Its maintenance adapter keeps that decision out of the CLI layer. The
process-wide open interceptor also covers worker threads, while an existing
daemon lease must still validate its thread/task and archive binding.

The CLI never arms or suppresses process-wide lease enforcement on behalf of
one root: doing so would change authorization for every unrelated archive.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Lock

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

    The command owns a default-root configuration failure. Concrete writable
    opens still resolve their own archive and pass the per-open boundary.
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


@contextmanager
def cli_archive_writer_ownership() -> Iterator[None]:
    """Check each writable open and retain the owning root's exclusion."""
    from polylogue.core.write_lease import current_write_lease, require_write_lease
    from polylogue.maintenance.offline_guard import (
        archive_root_for_writable_tier,
        hold_daemon_start_exclusion,
        refuse_writable_tier_opens,
    )

    configured_root = _archive_root()
    if configured_root is not None:
        # Preserve the fail-closed platform check before executing the command.
        resident_archive_writer(configured_root)
    held_roots: set[Path] = set()
    ownership_lock = Lock()
    with ExitStack() as stack:

        def require_ownership(path: Path) -> None:
            root = archive_root_for_writable_tier(path, configured_root=configured_root)
            lease = current_write_lease()
            if lease is not None:
                require_write_lease(f"CLI writable tier open ({path})", archive_root=root)
                if lease.archive_root is not None:
                    return
            with ownership_lock:
                resident = resident_archive_writer(root)
                if resident is not None:
                    owned_root, reason = resident
                    raise ArchiveWriterOwnershipError(
                        f"this CLI process may not write {path}: {reason}. "
                        "Route the mutation through that archive's daemon",
                        archive_root=owned_root,
                        resident_writer=reason,
                    )
                if root not in held_roots:
                    # The nonblocking lock also closes the probe/open race.
                    stack.enter_context(hold_daemon_start_exclusion(root))
                    held_roots.add(root)

        with refuse_writable_tier_opens(require_ownership):
            yield

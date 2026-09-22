"""Backup CLI command."""

from __future__ import annotations

from pathlib import Path

import click

from polylogue.daemon.backup import BACKUP_PROFILES, BackupProfile, backup_archive, format_backup_result
from polylogue.logging import configure_logging


def _refuse_while_a_daemon_owns_the_archive() -> None:
    """Refuse a backup that would checkpoint tiers a resident daemon is writing.

    ``backup`` is a declared offline authority, and it is the one CLI writer
    the process-wide ownership boundary in
    :mod:`polylogue.cli.write_authority` cannot catch: ``backup_archive``
    takes its own ``write_lease("maintenance.backup")``
    (``polylogue/daemon/backup.py``), so ``require_write_lease`` is satisfied
    by this process and the armed guard lets the write through. The snapshot
    is not a read -- ``_backup_sqlite`` opens each live tier through
    ``open_isolated_write_connection``, drains its WAL with a ``TRUNCATE``
    checkpoint and holds ``BEGIN IMMEDIATE`` across the copy -- and its own
    contract says so: "an exclusive boundary that already requires no
    concurrent writer". Nothing established that, so a backup run beside
    ``polylogued run`` truncated the daemon's WAL underneath it and retried
    ``_SNAPSHOT_LOCK_ATTEMPTS`` times to win the lock (polylogue-5vps8 AC1,
    polylogue-re6s3 AC1).

    ``--check`` is deliberately not guarded: it opens nothing writable, and a
    prerequisite check is exactly what an operator runs *before* stopping the
    daemon.
    """
    from polylogue.cli.write_authority import ArchiveWriterOwnershipError, resident_archive_writer

    resident = resident_archive_writer()
    if resident is None:
        return
    owned_root, reason = resident
    raise ArchiveWriterOwnershipError(
        f"refusing to back up {owned_root}: {reason}. A backup snapshot checkpoints and "
        "write-locks each live tier, so it must own the archive exclusively. Stop polylogued "
        "and run the backup again, or run `polylogue ops backup --check` to verify "
        "prerequisites without touching the tiers",
        archive_root=owned_root,
        resident_writer=reason,
    )


@click.command("backup", help="Back up the Polylogue archive durability tiers.")
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to write the backup into.",
)
@click.option(
    "--check",
    "check_only",
    is_flag=True,
    default=False,
    help="Verify backup prerequisites without creating a backup.",
)
@click.option(
    "--verify",
    "verify",
    is_flag=True,
    default=False,
    help="Restore the finished backup into a scratch directory and run smoke checks.",
)
@click.option(
    "--profile",
    type=click.Choice(BACKUP_PROFILES),
    default="rebuildable_cache_exclude",
    show_default=True,
    help="Named backup profile controlling copied archive tiers.",
)
def backup_command(
    output_dir: Path,
    check_only: bool,
    verify: bool,
    profile: BackupProfile,
) -> None:
    """Back up the Polylogue archive.

    Backups copy the precious source/user/embeddings tiers
    plus referenced blobs, and intentionally omit rebuildable index.db and
    disposable ops.db.

    Use --check to verify prerequisites (disk space, DB readability)
    without creating a backup.
    """
    configure_logging()
    if not check_only:
        _refuse_while_a_daemon_owns_the_archive()
    result = backup_archive(
        output_dir=output_dir,
        check_only=check_only,
        verify=verify,
        profile=profile,
    )
    for line in format_backup_result(result):
        click.echo(line)
    if not result.ok:
        raise SystemExit(1)


__all__ = ["backup_command"]

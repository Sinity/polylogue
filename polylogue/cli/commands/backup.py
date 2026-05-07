"""Backup CLI command."""

from __future__ import annotations

from pathlib import Path

import click

from polylogue.daemon.backup import backup_archive, format_backup_result
from polylogue.logging import configure_logging


@click.command("backup", help="Back up the Polylogue archive database.")
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
    "--include-blobs",
    "include_blobs",
    is_flag=True,
    default=False,
    help="Also back up the blob store directory.",
)
def backup_command(
    output_dir: Path,
    check_only: bool,
    include_blobs: bool,
) -> None:
    """Back up the Polylogue archive database and optionally the blob store.

    Uses SQLite VACUUM INTO for a clean, defragmented copy when available
    (SQLite >= 3.27.0), falling back to a file copy with WAL checkpoint.

    Use --check to verify prerequisites (disk space, DB readability)
    without creating a backup.
    """
    configure_logging()
    result = backup_archive(
        output_dir=output_dir,
        check_only=check_only,
        include_blobs=include_blobs,
    )
    for line in format_backup_result(result):
        click.echo(line)
    if not result.ok:
        raise SystemExit(1)


__all__ = ["backup_command"]

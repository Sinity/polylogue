"""Defensive ``archive.db`` → ``polylogue.db`` symlink at the archive root.

A historical rename left a 0-byte ``archive.db`` stub at the archive
root on some deployments (#1627).  Any legacy consumer that still
computes ``<archive_root>/archive.db`` should open the canonical
``polylogue.db`` instead.

This module provides a single idempotent helper that replaces a
0-byte stub with a symlink and warns about a non-empty legacy file.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.logging import get_logger

logger = get_logger(__name__)


def ensure_canonical_archive_db_name(db_path: Path) -> None:
    """Replace a stale ``archive.db`` stub with a symlink to the real database.

    *db_path* is the *absolute* path to the canonical ``polylogue.db`` file.
    The helper checks for ``archive.db`` in the same directory and:

    - If it is a 0-byte regular file, replace it with a symlink to
      ``db_path.name`` (i.e. ``polylogue.db``).
    - If it is a non-empty regular file, log a warning and leave it alone.
    - If it is already a symlink (or anything else), do nothing.
    """
    archive_db = db_path.parent / "archive.db"
    if not archive_db.exists():
        return

    if archive_db.is_symlink():
        return

    if not archive_db.is_file():
        return

    try:
        size = archive_db.stat().st_size
    except OSError:
        return

    if size == 0:
        try:
            archive_db.unlink()
            archive_db.symlink_to(db_path.name)
            logger.info(
                "archive_db_stub: replaced 0-byte %s with symlink to %s",
                archive_db,
                db_path.name,
            )
        except OSError:
            logger.warning("archive_db_stub: could not replace stub %s", archive_db, exc_info=True)
    else:
        logger.warning(
            "archive_db_stub: %s is a non-empty file (%d bytes); inspect before removing",
            archive_db,
            size,
        )


__all__ = ["ensure_canonical_archive_db_name"]

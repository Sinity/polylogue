"""Backup and portability operations for the Polylogue archive.

Provides a local-first backup command that copies the archive database
and (optionally) the blob store to a target directory.

Uses SQLite VACUUM INTO when available (SQLite >= 3.27.0) for a clean,
defragmented copy. Falls back to a file copy with WAL checkpoint first.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from polylogue.logging import get_logger
from polylogue.paths import blob_store_root, db_path

logger = get_logger(__name__)


class BackupResult(BaseModel):
    """Result of a backup operation."""

    ok: bool
    output_path: str | None = None
    db_size_bytes: int = 0
    blob_count: int = 0
    blob_size_bytes: int = 0
    elapsed_s: float = 0.0
    error: str | None = None
    check_only: bool = False
    warnings: list[str] = []


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _backup_db_vacuum_into(src: Path, dst: Path) -> None:
    """Backup using VACUUM INTO (SQLite >= 3.27.0)."""
    conn = sqlite3.connect(str(src))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute(f"VACUUM INTO '{dst}'")
    finally:
        conn.close()


def _backup_db_copy(src: Path, dst: Path) -> None:
    """Backup using file copy after WAL checkpoint."""
    conn = sqlite3.connect(str(src))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    shutil.copy2(src, dst)


def _check_prerequisites() -> list[str]:
    """Return a list of warning/error strings for backup prerequisites."""
    warnings: list[str] = []

    dbf = db_path()
    if not dbf.exists():
        return [f"database not found at {dbf}"]

    # Check database is readable.
    try:
        conn = sqlite3.connect(str(dbf))
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
        conn.close()
    except sqlite3.Error as exc:
        return [f"database not readable: {exc}"]

    # Check available disk space (rough estimate: 2x db size for VACUUM INTO).
    try:
        db_size = dbf.stat().st_size
        wal = dbf.with_suffix(".db-wal")
        if wal.exists():
            db_size += wal.stat().st_size
        # Estimate need ~2.5x for VACUUM INTO overhead.
        needed = int(db_size * 2.5)
        import os

        st = os.statvfs(str(dbf.parent))
        free = st.f_frsize * st.f_bavail
        if free < needed:
            warnings.append(
                f"low disk space: {free / (1024**3):.1f} GB free, ~{needed / (1024**3):.1f} GB needed for VACUUM INTO"
            )
    except Exception:
        pass

    return warnings


def backup_archive(
    *,
    output_dir: Path,
    check_only: bool = False,
    include_blobs: bool = False,
) -> BackupResult:
    """Backup the Polylogue archive database and optionally the blob store.

    Args:
        output_dir: Target directory for the backup.
        check_only: If True, only verify prerequisites without creating a backup.
        include_blobs: Also copy the blob store directory.

    Returns:
        ``BackupResult`` with outcome details.
    """
    started = time.monotonic()

    if check_only:
        warnings = _check_prerequisites()
        return BackupResult(
            ok=len(warnings) == 0,
            check_only=True,
            warnings=warnings,
            error=warnings[0] if warnings else None,
            elapsed_s=round(time.monotonic() - started, 3),
        )

    # Non-check mode: actually create backup.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dbf = db_path()
    if not dbf.exists():
        return BackupResult(
            ok=False,
            error=f"database not found at {dbf}",
            elapsed_s=round(time.monotonic() - started, 3),
        )

    warnings = _check_prerequisites()
    has_errors = any("not found" in w or "not readable" in w for w in warnings)

    ts = _timestamp()
    db_dst = output_dir / f"polylogue-{ts}.db"
    db_size = 0
    blob_count = 0
    blob_size = 0

    logger.info("backup: starting archive backup to %s", output_dir)

    # Back up the database.
    try:
        _backup_db_vacuum_into(dbf, db_dst)
        if db_dst.exists():
            db_size = db_dst.stat().st_size
        logger.info("backup: database written to %s (%d bytes)", db_dst, db_size)
    except sqlite3.OperationalError:
        # VACUUM INTO not supported — fall back to copy.
        logger.info("backup: VACUUM INTO not available, falling back to file copy")
        _backup_db_copy(dbf, db_dst)
        if db_dst.exists():
            db_size = db_dst.stat().st_size
        logger.info("backup: database copied to %s (%d bytes)", db_dst, db_size)

    # Back up blob store if requested.
    if include_blobs:
        blob_root = blob_store_root()
        if blob_root.exists():
            blob_dst = output_dir / f"blob-{ts}"
            try:
                shutil.copytree(blob_root, blob_dst, symlinks=True)
                for f in blob_dst.rglob("*"):
                    if f.is_file():
                        blob_count += 1
                        blob_size += f.stat().st_size
                logger.info(
                    "backup: blob store copied to %s (%d blobs, %d bytes)",
                    blob_dst,
                    blob_count,
                    blob_size,
                )
            except Exception as exc:
                warnings.append(f"blob store backup failed: {exc}")
                logger.warning("backup: blob store copy failed: %s", exc)

    elapsed = round(time.monotonic() - started, 3)
    ok = not has_errors

    return BackupResult(
        ok=ok,
        output_path=str(db_dst),
        db_size_bytes=db_size,
        blob_count=blob_count,
        blob_size_bytes=blob_size,
        elapsed_s=elapsed,
        warnings=warnings,
    )


def format_backup_result(result: BackupResult) -> list[str]:
    """Render backup result as plain-text lines."""
    lines: list[str] = []
    if result.check_only:
        if result.ok:
            lines.append("Backup prerequisites: OK")
        else:
            lines.append(f"Backup prerequisites: FAILED — {result.error}")
        for w in result.warnings:
            lines.append(f"  Warning: {w}")
        return lines

    if result.ok:
        lines.append(f"Backup complete: {result.output_path}")
    else:
        lines.append(f"Backup failed: {result.error}")
        lines.append(f"  Partial output: {result.output_path}")

    lines.append(f"  DB size: {result.db_size_bytes / (1024**2):.1f} MB")
    if result.blob_count:
        lines.append(f"  Blobs: {result.blob_count} ({result.blob_size_bytes / (1024**2):.1f} MB)")
    lines.append(f"  Elapsed: {result.elapsed_s:.1f}s")
    for w in result.warnings:
        lines.append(f"  Warning: {w}")
    return lines


__all__ = [
    "BackupResult",
    "backup_archive",
    "format_backup_result",
]

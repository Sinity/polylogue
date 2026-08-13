"""Immutable archive templates for SQLite-heavy real-route tests."""

from __future__ import annotations

import contextlib
import gc
import shutil
import sqlite3
import stat
import subprocess
import time
from pathlib import Path


def _journal_mode_delete_with_retry(conn: sqlite3.Connection, *, path: Path) -> None:
    """Finish a SQLite snapshot despite a deferred same-process close."""
    deadline = time.monotonic() + 5.0
    attempt = 0
    while True:
        attempt += 1
        try:
            mode = conn.execute("PRAGMA journal_mode=DELETE").fetchone()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                raise
            gc.collect()
            time.sleep(min(0.05 * attempt, 0.5))
            continue
        if mode != ("delete",):
            raise RuntimeError(f"could not normalize template database {path} to DELETE journal mode")
        return


def quiesce_archive_template(root: Path) -> None:
    """Validate every physical SQLite tier and collapse it into one snapshot."""
    databases = tuple(path for path in sorted(root.rglob("*.db")) if path.is_file() and not path.is_symlink())
    for path in databases:
        with contextlib.closing(sqlite3.connect(path)) as conn, conn:
            quick = conn.execute("PRAGMA quick_check").fetchone()
            foreign = conn.execute("PRAGMA foreign_key_check").fetchall()
            if quick != ("ok",) or foreign:
                raise RuntimeError(f"invalid template database {path}")
            checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint is None or checkpoint[0] != 0:
                raise RuntimeError(f"could not checkpoint template database {path}: {checkpoint!r}")
            _journal_mode_delete_with_retry(conn, path=path)
        for suffix in ("-wal", "-shm"):
            sidecar = path.with_name(f"{path.name}{suffix}")
            if sidecar.exists():
                sidecar.unlink()


def _freeze_archive_template(root: Path) -> None:
    """Make a completed fixture archive immutable before test-local cloning."""
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    root.chmod(root.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def finalize_archive_template(root: Path) -> None:
    """Publish a reusable archive template only after a verified SQLite snapshot."""
    quiesce_archive_template(root)
    _freeze_archive_template(root)


def clone_archive_template(template: Path, destination: Path) -> None:
    """Clone an immutable archive into a private writable destination."""
    destination.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["cp", "-a", "--reflink=auto", f"{template}/.", str(destination)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        shutil.copytree(template, destination, dirs_exist_ok=True, symlinks=True)
    for path in destination.rglob("*"):
        if path.is_symlink():
            continue
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    destination.chmod(destination.stat().st_mode | stat.S_IWUSR)
    bootstrap_marker = destination / ".maintenance-state" / "durable-change-trains" / ".bootstrap"
    if bootstrap_marker.is_file():
        from polylogue.storage.sqlite.durable_change_train import _record_fresh_durable_bootstrap

        bootstrap_marker.unlink()
        _record_fresh_durable_bootstrap(destination)


__all__ = ["clone_archive_template", "finalize_archive_template", "quiesce_archive_template"]

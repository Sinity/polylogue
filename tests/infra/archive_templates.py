"""Immutable archive templates for SQLite-heavy real-route tests."""

from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path


def freeze_archive_template(root: Path) -> None:
    """Make a completed fixture archive immutable before test-local cloning."""
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    root.chmod(root.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


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


__all__ = ["clone_archive_template", "freeze_archive_template"]

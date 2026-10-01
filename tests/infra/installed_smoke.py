"""Synthetic executable fixtures for the installed smoke lifecycle."""

from __future__ import annotations

import shutil
from pathlib import Path


def make_installed_smoke_bins(directory: Path) -> Path:
    source = Path(__file__).resolve().parents[1] / "fixtures/installed_smoke"
    bins = directory / "bin"
    bins.mkdir()
    for name, fixture in [
        ("python", "python.py"),
        ("polylogue", "cli.py"),
        ("polylogued", "daemon.py"),
        ("polylogue-mcp", "cli.py"),
    ]:
        target = bins / name
        shutil.copyfile(source / fixture, target)
        target.chmod(0o755)
    return bins

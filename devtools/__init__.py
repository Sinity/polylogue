"""Developer tools for polylogue project maintenance."""

from __future__ import annotations

import subprocess
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root, correct from any git worktree.

    Uses ``git rev-parse --show-toplevel`` so that the returned path is
    the worktree root rather than the main checkout when invoked from
    inside a git worktree.  Falls back to resolving from ``__file__``
    if git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return Path(__file__).resolve().parent.parent

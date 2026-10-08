"""The shared containment policy for excluded regular-file aliases."""

from __future__ import annotations

import stat
from pathlib import Path


def contained_file_alias_coordinate(physical_root: Path, target: Path, mode: int) -> str | None:
    """Name a regular target inside the captured physical root, or refuse it.

    Callers own target identity and no-follow opening. Containment alone does
    not prove that the target has an independently observed material row.
    """
    if not stat.S_ISREG(mode) or not target.is_relative_to(physical_root):
        return None
    return target.relative_to(physical_root).as_posix()

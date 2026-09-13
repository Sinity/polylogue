"""Read seams for bounded diagnostic probes used by surface adapters."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


@contextmanager
def one_shot_diagnostic_read(
    path: str | Path,
    *,
    tier: ArchiveTier | None = None,
) -> Iterator[sqlite3.Connection]:
    """Yield one bounded diagnostic read through the storage connection profile.

    Surface adapters use this seam instead of importing the storage profile
    directly.  Resolve the profile module at call time so storage-level test
    doubles remain observable at the ownership boundary.
    """
    from polylogue.storage.sqlite import connection_profile

    with connection_profile.one_shot_diagnostic_read(path, tier=tier) as conn:
        yield conn


__all__ = ["one_shot_diagnostic_read"]

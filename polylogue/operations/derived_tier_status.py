"""Read-only derived-tier identity probes for operation-owned status surfaces."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.schema_identity import (
    DerivedTier,
    derived_schema_identity,
    read_schema_identity,
)
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


def derived_tier_identity(path: Path, tier_name: str) -> tuple[str, str | None]:
    """Read expected and stamped identity without creating a writer path."""

    tier = DerivedTier(tier_name)
    expected = derived_schema_identity(tier)
    try:
        with closing(open_readonly_connection(path, validate_schema=False)) as conn:
            actual = read_schema_identity(conn, tier)
    except sqlite3.Error as exc:
        # An unreadable tier cannot serve this runtime; type it here so callers
        # classify one ValueError instead of each catching sqlite3 errors.
        raise ValueError(f"derived tier {tier_name} is unreadable: {exc}") from exc
    return expected, actual


def derived_identity_mismatches(tier_paths: Mapping[str, Path]) -> list[str]:
    """Return derived tiers whose stamped identity cannot serve this runtime."""

    mismatches: list[str] = []
    for name in ("index", "ops"):
        path = tier_paths[name]
        if not path.exists():
            continue
        try:
            expected, actual = derived_tier_identity(path, name)
        except (OSError, ValueError):
            mismatches.append(name)
            continue
        if actual != expected:
            mismatches.append(name)
    return mismatches


__all__ = ["derived_identity_mismatches", "derived_tier_identity"]

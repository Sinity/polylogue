"""Acquire an archive tier read handle, or a typed refusal carrying the probe.

Tier absence, schema skew and an unopenable file are classified once, here,
against ``ArchiveTierProbe``. A caller receives ``TierHandle`` (which owns the
connection) or ``TierRefusal`` (which owns no connection at all), so reaching
SQL without first handling absence is a type error rather than a convention.
That is the whole point: the refusal carries ``version_status`` from the probe,
so a read path reports *why* it has no rows instead of improvising an empty
result behind its own ``except sqlite3``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.evidence import Evidence, Measured, Unavailable
from polylogue.storage.archive_readiness import ArchiveTierProbe, probe_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

__all__ = [
    "TierHandle",
    "TierRefusal",
    "acquire_tier_reader",
    "open_tier_reader",
    "tier_evidence",
]


@dataclass(frozen=True, slots=True)
class TierHandle:
    """An open read-only connection to one tier, with the probe that admitted it."""

    tier: ArchiveTier
    path: Path
    connection: sqlite3.Connection
    probe: ArchiveTierProbe


@dataclass(frozen=True, slots=True)
class TierRefusal:
    """Why one tier could not be read. Carries no connection, by construction."""

    tier: ArchiveTier
    path: Path
    version_status: str
    reason: str
    detail: str | None = None


def acquire_tier_reader(tier: ArchiveTier, path: Path) -> TierHandle | TierRefusal:
    """Open ``path`` for reading, or refuse with the probe's status.

    The caller owns ``TierHandle.connection`` and must close it; use
    ``open_tier_reader`` where a scope owns the handle.
    """
    probe = probe_archive_tier(tier, path)
    if not probe.exists:
        return TierRefusal(
            tier=tier,
            path=path,
            version_status=probe.version_status,
            reason="tier_missing",
            detail=f"{path} does not exist",
        )
    if probe.version_status == "mismatch":
        return TierRefusal(
            tier=tier,
            path=path,
            version_status=probe.version_status,
            reason="schema_version_mismatch",
            detail=f"user_version={probe.user_version} expected={probe.expected_user_version}",
        )
    if probe.version_status == "invalid":
        return TierRefusal(
            tier=tier,
            path=path,
            version_status=probe.version_status,
            reason="tier_unreadable",
            detail=f"{path} could not be opened for a version probe",
        )
    try:
        connection = open_readonly_connection(path)
    except sqlite3.Error as exc:
        return TierRefusal(
            tier=tier,
            path=path,
            version_status=probe.version_status,
            reason="tier_unreadable",
            detail=f"{type(exc).__name__}: {exc}",
        )
    return TierHandle(tier=tier, path=path, connection=connection, probe=probe)


@contextmanager
def open_tier_reader(tier: ArchiveTier, path: Path) -> Iterator[TierHandle | TierRefusal]:
    """Scope-own the handle; a refusal yields with nothing to close."""
    acquired = acquire_tier_reader(tier, path)
    try:
        yield acquired
    finally:
        if isinstance(acquired, TierHandle):
            acquired.connection.close()


def tier_evidence(acquired: TierHandle | TierRefusal) -> Evidence[TierHandle]:
    """Lift an acquisition into the shared evidence union for payload assembly."""
    if isinstance(acquired, TierHandle):
        return Measured(acquired)
    return Unavailable(reason=acquired.reason, detail=acquired.detail)

"""Tier acquisition hands back a connection or a reason, never a silent empty.

Anti-vacuity: give ``TierRefusal`` a ``connection`` attribute, or make
``acquire_tier_reader`` return a handle for a missing or version-skewed tier,
and these assertions fail.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.core.evidence import Measured, Unavailable
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.tier_access import TierHandle, TierRefusal, acquire_tier_reader, open_tier_reader, tier_evidence


def _index_path(root: Path) -> Path:
    return root / ARCHIVE_TIER_SPECS[ArchiveTier.INDEX].filename


def _initialize_tiers(root: Path) -> None:
    for tier, spec in ARCHIVE_TIER_SPECS.items():
        initialize_archive_database(root / spec.filename, tier)


def test_missing_tier_refuses_with_the_probe_status(tmp_path: Path) -> None:
    acquired = acquire_tier_reader(ArchiveTier.INDEX, _index_path(tmp_path))

    assert isinstance(acquired, TierRefusal)
    assert acquired.reason == "tier_missing"
    assert acquired.version_status == "missing"
    assert not hasattr(acquired, "connection")


def test_version_skew_refuses_instead_of_reading(tmp_path: Path) -> None:
    _initialize_tiers(tmp_path)
    index_path = _index_path(tmp_path)
    skewed = ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX] + 1
    with sqlite3.connect(index_path) as conn:
        conn.execute(f"PRAGMA user_version = {skewed}")

    acquired = acquire_tier_reader(ArchiveTier.INDEX, index_path)

    assert isinstance(acquired, TierRefusal)
    assert acquired.reason == "schema_version_mismatch"
    assert acquired.version_status == "mismatch"
    assert str(skewed) in (acquired.detail or "")


def test_healthy_tier_yields_a_readable_handle(tmp_path: Path) -> None:
    _initialize_tiers(tmp_path)

    with open_tier_reader(ArchiveTier.INDEX, _index_path(tmp_path)) as acquired:
        assert isinstance(acquired, TierHandle)
        assert acquired.probe.version_status == "ok"
        assert acquired.connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_acquisition_lifts_into_the_shared_evidence_union(tmp_path: Path) -> None:
    refused = tier_evidence(acquire_tier_reader(ArchiveTier.INDEX, _index_path(tmp_path)))
    assert isinstance(refused, Unavailable)
    assert refused.reason == "tier_missing"

    _initialize_tiers(tmp_path)
    with open_tier_reader(ArchiveTier.INDEX, _index_path(tmp_path)) as acquired:
        assert isinstance(tier_evidence(acquired), Measured)

"""A durable tier below the version this archive recorded at birth is not its tier.

The format marker records the durable version each tier was bootstrapped at.
``assert_archive_format_lineage`` compared the live file against the lineage
*floor* and only checked the recorded schema fingerprint when the live version
*equalled* the recorded birth version -- so once a tier is born above the floor
(the source tier is, at numbered slot 002), a transplanted historical version-1
file cleared the floor, skipped the fingerprint on the version mismatch, and
was admitted as lineage evidence before bootstrap, migration or backup
verification trusted it. The fingerprint now also decides the below-birth case,
which keeps this lineage's own tier -- one numbered slot behind, carrying the
recorded schema -- on the migration route rather than refusing it as foreign.

Anti-vacuity: narrow the fingerprint condition back to ``version ==
versions[tier.value]`` in
``polylogue/storage/sqlite/archive_tiers/archive_plan.py`` and
``test_transplanted_older_tier_is_refused`` goes red -- the transplanted
version-1 source tier is accepted with no error at all.
``test_freshly_bootstrapped_archive_is_accepted`` and
``test_a_migrated_tier_above_its_birth_version_is_accepted`` pin the opposite
direction, so refusing everything cannot pass.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.archive_plan import assert_archive_format_lineage
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _bootstrap(root: Path) -> None:
    with ArchiveStore(root):
        pass


def _stamp(path: Path, version: int) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(f"PRAGMA user_version = {int(version)}")
        conn.commit()
    finally:
        conn.close()


class TestArchiveFormatBirthVersion:
    def test_freshly_bootstrapped_archive_is_accepted(self, tmp_path: Path) -> None:
        """Opposite direction: an archive must never refuse its own marker."""
        _bootstrap(tmp_path)
        assert_archive_format_lineage(tmp_path)

    def test_transplanted_older_tier_is_refused(self, tmp_path: Path) -> None:
        """A tier born at version 2 cannot be a version-1 file from another lineage."""
        source_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
        assert source_version > 1, "this archive lineage no longer births its source tier above the floor"
        _bootstrap(tmp_path)

        source_db = tmp_path / "source.db"
        source_db.unlink()
        conn = sqlite3.connect(source_db)
        try:
            # A historical lineage's shape, stamped with the pre-reset integer.
            conn.execute("CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY, blob_hash BLOB)")
            conn.execute("PRAGMA user_version = 1")
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(RuntimeError, match="historical version-1 schema"):
            assert_archive_format_lineage(tmp_path)

    def test_a_migrated_tier_above_its_birth_version_is_accepted(self, tmp_path: Path) -> None:
        """Opposite direction: a tier a later migration raised must stay admissible.

        The marker is a lower bound, not a pin: requiring equality would make
        every archive refuse itself the moment the next numbered migration
        lands.
        """
        _bootstrap(tmp_path)
        _stamp(tmp_path / "source.db", ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] + 1)
        assert_archive_format_lineage(tmp_path)

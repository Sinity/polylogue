"""An old v1 tier is not admitted into the v2 archive lineage.

The prior v1 lineage also stamped source.db with version 1, so the v2 marker
must use canonical schema fingerprints to refuse that old shape.

Anti-vacuity: narrowing the fingerprint condition back to an exact birth-version
comparison admits this transplanted source tier without an error.
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
        """A foreign shape with the same version integer still fails fingerprint validation."""
        _bootstrap(tmp_path)

        source_db = tmp_path / "source.db"
        source_db.unlink()
        conn = sqlite3.connect(source_db)
        try:
            # The old lineage used integer 1 too, but declared a different schema.
            conn.execute("CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY, blob_hash BLOB)")
            conn.execute("PRAGMA user_version = 1")
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(RuntimeError, match="historical version-1 schema"):
            assert_archive_format_lineage(tmp_path)

    def test_a_future_tier_version_above_its_birth_version_is_accepted(self, tmp_path: Path) -> None:
        """Opposite direction: a later version in this format lineage stays admissible.

        The marker is a lower bound, not a pin: requiring equality would make
        every archive refuse itself the moment the next numbered migration
        lands.
        """
        _bootstrap(tmp_path)
        _stamp(tmp_path / "source.db", ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] + 1)
        assert_archive_format_lineage(tmp_path)

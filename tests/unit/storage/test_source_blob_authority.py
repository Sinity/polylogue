"""A ``user_version`` stamp alone cannot make a source tier canonical.

``user_version`` was renumbered from one by the archive format floor, so it no
longer orders across lineages: a historical source database can carry exactly
the integer this runtime stamps today while holding only the legacy
``raw_sessions(raw_id, blob_hash)`` carrier. ``_source_schema_capabilities``
granted such a file ``current_authority`` on the integer alone, which sent
``referenced_blob_hashes(..., require_index=False)`` into the blocked canonical
query instead of the historical fallback -- making existing blob references
unavailable to integrity and recovery tooling. This projection reads a bare
connection and has no format marker to separate the lineages, so the catalog
has to earn the authority: ``blob_refs`` is the typed ledger only this lineage
writes.

Anti-vacuity: restore ``user_version == stamped_source_version or
current_capabilities`` in ``polylogue/storage/blob_integrity.py`` and
``test_legacy_carrier_keeps_its_fallback_at_the_colliding_stamp`` goes red --
the capability projection reports ``current_authority`` and the reference read
raises instead of returning the archive's one legacy hash.
``test_a_real_current_source_tier_is_still_canonical`` pins the opposite
direction so "never trust the stamp" alone cannot pass.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.blob_integrity import _source_schema_capabilities, referenced_blob_hashes
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_LEGACY_HASH = bytes(range(32))


def _legacy_source_db(path: Path, *, stamp: int) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY, blob_hash BLOB)")
        conn.execute("INSERT INTO raw_sessions VALUES ('r1', ?)", (_LEGACY_HASH,))
        conn.execute(f"PRAGMA user_version = {int(stamp)}")
        conn.commit()
    finally:
        conn.close()


class TestSourceBlobAuthority:
    @pytest.mark.parametrize("stamp_offset", [0, -1, 27])
    def test_legacy_carrier_keeps_its_fallback_at_the_colliding_stamp(self, tmp_path: Path, stamp_offset: int) -> None:
        """Every stamp a historical file can carry keeps the catalog classification.

        ``stamp_offset == 0`` is the collision that matters: the integer this
        runtime stamps today, on a file this runtime never wrote.
        """
        stamped = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
        source_db = tmp_path / "source.db"
        _legacy_source_db(source_db, stamp=max(1, stamped + stamp_offset))

        conn = sqlite3.connect(source_db)
        try:
            capabilities = _source_schema_capabilities(conn)
        finally:
            conn.close()
        assert capabilities.current_authority is False
        assert capabilities.legacy_carriers == ("raw_sessions",)

        assert set(referenced_blob_hashes(source_db, require_index=False)) == {_LEGACY_HASH.hex()}

    def test_a_real_current_source_tier_is_still_canonical(self, tmp_path: Path) -> None:
        """Opposite direction: a bootstrapped source tier keeps canonical authority."""
        with ArchiveStore(tmp_path):
            pass
        conn = sqlite3.connect(tmp_path / "source.db")
        try:
            capabilities = _source_schema_capabilities(conn)
        finally:
            conn.close()
        assert capabilities.current_authority is True
        assert capabilities.kind == "current_versioned"

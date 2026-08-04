from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.daemon.backup import _all_archive_tiers, _profile_archive_tiers
from polylogue.storage.archive_identity import ArchiveIdentity
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import DURABLE_MIGRATION_TIERS


def test_audit_tier_bootstrap_has_v1_authority_tables_and_is_durable(tmp_path: Path) -> None:
    path = tmp_path / "audit.db"
    initialize_archive_database(path, ArchiveTier.AUDIT)
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()
    assert {
        "archive_authority",
        "operation_previews",
        "operation_authorizations",
        "operation_runs",
        "operation_events",
    } <= tables
    assert ArchiveTier.AUDIT in DURABLE_MIGRATION_TIERS
    assert ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT] == ARCHIVE_TIER_SPECS[ArchiveTier.AUDIT].version
    assert ARCHIVE_TIER_SPECS[ArchiveTier.AUDIT].backup_required is True


def test_audit_is_in_authority_backup_inventory_but_not_its_own_identity_digest(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    before = ArchiveIdentity.resolve(root).authority_identity_digest
    (root / "audit.db").touch()
    after = ArchiveIdentity.resolve(root).authority_identity_digest
    assert before == after
    all_tiers = _all_archive_tiers(root)
    assert all_tiers["audit"] == root / "audit.db"
    assert "audit" in _profile_archive_tiers(root, "full_evidence")
    assert "audit" in _profile_archive_tiers(root, "user_overlays")
    assert "audit" in _profile_archive_tiers(root, "rebuildable_cache_exclude")

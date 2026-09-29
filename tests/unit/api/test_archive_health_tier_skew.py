"""Archive health reports a tier its runtime refuses to serve instead of raising."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue import Polylogue
from polylogue.readiness import ReadinessReport, VerifyStatus
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_templates import bootstrap_archive_root


async def _health_check(root: Path) -> ReadinessReport:
    archive = Polylogue(archive_root=root, db_path=root / "index.db")
    try:
        return await archive.health_check()
    finally:
        await archive.close()


async def test_health_check_reports_durable_version_skew(tmp_path: Path) -> None:
    """Fails if the tier probe lets the open's SchemaSkew escape ``health_check``."""
    bootstrap_archive_root(tmp_path)
    expected = ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT]
    mismatched = expected + 1
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        conn.execute(f"PRAGMA user_version = {mismatched}")

    report = await _health_check(tmp_path)

    check = next(item for item in report.checks if item.name == "archive_audit")
    assert check.status == VerifyStatus.ERROR
    assert f"found {mismatched}" in (check.summary or "")
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == mismatched


async def test_health_check_reports_derived_identity_skew(tmp_path: Path) -> None:
    """Fails if the probe skips schema admission: a version-only comparison reports this tier OK."""
    bootstrap_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("UPDATE schema_identity SET identity = 'stale-identity' WHERE tier = 'ops'")
        assert conn.execute("PRAGMA user_version").fetchone()[0] == ARCHIVE_VERSION_BY_TIER[ArchiveTier.OPS]

    report = await _health_check(tmp_path)

    check = next(item for item in report.checks if item.name == "archive_ops")
    assert check.status == VerifyStatus.ERROR
    assert "stale-identity" in (check.summary or "")

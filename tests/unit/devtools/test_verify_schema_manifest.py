from __future__ import annotations

import sqlite3
from pathlib import Path

from devtools import verify_schema_manifest
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_schema_manifest_checks_all_canonical_tiers() -> None:
    assert verify_schema_manifest.main([]) == 0


def test_schema_manifest_rejects_a_target_file_with_schema_drift(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    for tier in ArchiveTier:
        if tier is ArchiveTier.EMBEDDINGS:
            continue
        initialize_archive_database(root / f"{tier.value}.db", tier)
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute("DROP INDEX idx_sessions_origin")
        conn.commit()
    assert verify_schema_manifest.main(["--archive-root", str(root)]) == 1

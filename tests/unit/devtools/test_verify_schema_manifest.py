from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from devtools import verify_schema_manifest
from polylogue.storage.archive_identity import ArchiveLocation
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


def test_schema_manifest_rejects_a_missing_explicit_archive_tier(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    for tier in ArchiveTier:
        if tier is not ArchiveTier.EMBEDDINGS:
            initialize_archive_database(root / f"{tier.value}.db", tier)
    (root / "user.db").unlink()

    assert verify_schema_manifest.main(["--archive-root", str(root)]) == 1


def test_schema_manifest_opens_special_character_paths_read_only(tmp_path: Path) -> None:
    path = tmp_path / "index?with#fragment.db"
    initialize_archive_database(path, ArchiveTier.INDEX)

    assert verify_schema_manifest._check_tier(ArchiveTier.INDEX, path)["ok"] is True


def test_schema_manifest_checks_the_promoted_active_index(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    active = root / ".index-generations" / "generation" / "index.db"
    active.parent.mkdir(parents=True)
    initialize_archive_database(active, ArchiveTier.INDEX)
    (root / "index.db").write_bytes(b"stale shadow")
    (root / ".index-active-pointer").write_text(str(active), encoding="utf-8")

    result = verify_schema_manifest._check_tier(ArchiveTier.INDEX, ArchiveLocation.resolve(root).active_index_path)

    assert result["ok"] is True


def test_durable_ddl_evolution_requires_a_version_or_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "polylogue" / "storage" / "sqlite" / "archive_tiers" / "source.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        'SOURCE_SCHEMA_VERSION = 1\nSOURCE_DDL = """CREATE TABLE changed (id INTEGER)"""\n', encoding="utf-8"
    )
    old = 'SOURCE_SCHEMA_VERSION = 1\nSOURCE_DDL = """CREATE TABLE original (id INTEGER)"""\n'

    def fake_git_text(*args: str) -> str:
        if args[:2] == ("rev-parse", "--verify"):
            return "base\n"
        if args[:1] == ("show",):
            return old
        return ""

    monkeypatch.setattr(verify_schema_manifest, "ROOT", tmp_path)
    monkeypatch.setattr(verify_schema_manifest, "_git_text", fake_git_text)

    assert verify_schema_manifest._durable_ddl_evolution_violations() == [
        "source DDL changed without a schema-version bump or numbered migration"
    ]

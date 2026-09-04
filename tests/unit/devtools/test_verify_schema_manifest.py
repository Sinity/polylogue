from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

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
        conn.execute("DROP INDEX idx_sessions_origin_sort")
        conn.commit()
    assert verify_schema_manifest.main(["--archive-root", str(root)]) == 1


def _schema_state(*, source_version: int = 1, source_ddl: str = "source") -> verify_schema_manifest._SchemaState:
    ddl = {tier: tier.value for tier in ArchiveTier}
    ddl[ArchiveTier.SOURCE] = source_ddl
    versions = dict.fromkeys(ArchiveTier, 1)
    versions[ArchiveTier.SOURCE] = source_version
    return verify_schema_manifest._SchemaState(ddl=ddl, versions=versions)


def test_durable_evolution_requires_every_numbered_migration_for_a_version_bump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing v1 to v2 without adding 002 must make the gate red."""
    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(
        verify_schema_manifest,
        "_render_schema_state",
        lambda ref: _schema_state(source_version=1 if ref == "base" else 2),
    )
    monkeypatch.setattr(verify_schema_manifest, "_migration_changes", lambda _base, _tier: ())

    violations = verify_schema_manifest._durable_ddl_evolution_violations()

    assert any("source" in violation and "missing added migrations for v2" in violation for violation in violations)


def test_durable_evolution_compares_rendered_ddl_transformations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing rendered DDL through a post-assignment transform must make the gate red."""
    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(
        verify_schema_manifest,
        "_render_schema_state",
        lambda ref: _schema_state(source_ddl="before" if ref == "base" else "after"),
    )
    monkeypatch.setattr(verify_schema_manifest, "_migration_changes", lambda _base, _tier: ())

    violations = verify_schema_manifest._durable_ddl_evolution_violations()

    assert "source: rendered DDL changed without a schema-version bump" in violations


def test_durable_evolution_accepts_a_complete_contiguous_migration_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding 002 and 003 for a v1 to v3 bump must keep the gate green."""
    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(
        verify_schema_manifest,
        "_render_schema_state",
        lambda ref: _schema_state(source_version=1 if ref == "base" else 3),
    )
    changes = tuple(
        verify_schema_manifest._MigrationChange(
            "A",
            f"polylogue/storage/sqlite/migrations/source/{version:03d}_step.sql",
            f"polylogue/storage/sqlite/migrations/source/{version:03d}_step.sql",
        )
        for version in (2, 3)
    )
    monkeypatch.setattr(
        verify_schema_manifest,
        "_migration_changes",
        lambda _base, tier: changes if tier is ArchiveTier.SOURCE else (),
    )

    assert verify_schema_manifest._durable_ddl_evolution_violations() == []


def test_durable_evolution_fixture_covers_every_durable_tier() -> None:
    """Removing a durable tier from the fixture must make coverage fail."""
    assert set(verify_schema_manifest._DURABLE_TIERS) == {
        ArchiveTier.SOURCE,
        ArchiveTier.USER,
        ArchiveTier.AUDIT,
    }
    state = verify_schema_manifest._current_schema_state()
    assert all(tier in state.ddl and tier in state.versions for tier in verify_schema_manifest._DURABLE_TIERS)


def test_durable_evolution_uses_head_parent_when_origin_master_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkout without origin/master must still use its available parent base."""
    calls: list[tuple[str, ...]] = []

    def fake_git_text(*args: str) -> str:
        calls.append(args)
        if args == ("merge-base", "HEAD", "HEAD^"):
            return "parent\n"
        raise subprocess.CalledProcessError(1, ["git", *args])

    monkeypatch.setattr(verify_schema_manifest, "_git_text", fake_git_text)

    assert verify_schema_manifest._merge_base() == "parent"
    assert ("merge-base", "HEAD", "origin/master") in calls


@pytest.mark.parametrize("status", ["D", "M"])
def test_durable_evolution_rejects_deleted_or_modified_required_migrations(
    status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting or editing a required migration must remain independently visible."""
    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(verify_schema_manifest, "_render_schema_state", lambda _ref: _schema_state())
    change = verify_schema_manifest._MigrationChange(
        status,
        "polylogue/storage/sqlite/migrations/source/002_initial.sql",
        "polylogue/storage/sqlite/migrations/source/002_initial.sql",
    )
    monkeypatch.setattr(
        verify_schema_manifest,
        "_migration_changes",
        lambda _base, tier: (change,) if tier is ArchiveTier.SOURCE else (),
    )

    violations = verify_schema_manifest._durable_ddl_evolution_violations()

    assert any("source: required migration was" in violation for violation in violations)

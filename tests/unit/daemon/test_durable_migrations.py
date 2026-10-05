"""Declared durable migrations apply when the daemon opens the archive.

Each test ships a synthetic numbered migration package, raises the runtime's
declared tier version to it, and calls the daemon's open-time route under the
same exclusive archive ownership the daemon holds.

Anti-vacuity: return early from ``apply_declared_durable_migrations`` and the
tier stays at v1; drop ``pre_migration_backup`` from the data-changing path and
the train refuses with "requires an authenticated backup before apply".
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import AbstractContextManager
from pathlib import Path

import pytest

from polylogue.core.write_lease import write_lease
from polylogue.daemon.durable_migrations import apply_declared_durable_migrations
from polylogue.operations.durable_change_train import acquire_durable_archive_ownership
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.durable_tier_fixtures import bootstrap_baseline_archive


def _declare_future_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier: ArchiveTier,
    *,
    sql: str,
    new_table: str,
    base_ddl: str | None = None,
) -> None:
    _declare_future_migrations(tmp_path, monkeypatch, tier, steps=((sql, new_table),), base_ddl=base_ddl)


def _declare_future_migrations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier: ArchiveTier,
    *,
    steps: tuple[tuple[str, str], ...],
    base_ddl: str | None = None,
) -> None:
    """Ship numbered steps 002.. for ``tier`` and raise its declared version to match."""
    from polylogue.storage.sqlite import migration_runner
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER, bootstrap
    from polylogue.storage.sqlite.migration_runner import (
        DurableChangeRider,
        DurableRuntimeConsumer,
        declare_durable_change_train,
        durable_change_train_to_payload,
        durable_migration_claim_for_sql,
    )

    # A per-test package name: an imported package is cached in sys.modules.
    package = "fixture_open_migrations_" + "".join(ch if ch.isalnum() else "_" for ch in tmp_path.name)
    tier_package = tmp_path / package / tier.value
    tier_package.mkdir(parents=True)
    (tmp_path / package / "__init__.py").write_text("", encoding="utf-8")
    (tier_package / "__init__.py").write_text("", encoding="utf-8")
    for offset, (sql, new_table) in enumerate(steps):
        slot = 2 + offset
        name = f"{slot:03d}_{new_table}.sql"
        (tier_package / name).write_text(sql, encoding="utf-8")
        claim = durable_migration_claim_for_sql(tier, name, sql, owner_ref="owner:open")
        rider = DurableChangeRider(
            rider_id=f"rider:open-{slot}",
            owner_ref="owner:open-rider",
            schema_objects=(f"table:{new_table}",),
            runtime_consumers=(
                DurableRuntimeConsumer(
                    "bootstrap",
                    "polylogue/storage/sqlite/archive_tiers/bootstrap.py:initialize_archive_database",
                    "proof:bootstrap",
                    ("write",),
                ),
                DurableRuntimeConsumer(
                    "daemon-health",
                    "polylogue/storage/sqlite/archive_tiers/bootstrap.py:initialize_archive_tier",
                    "proof:daemon-health",
                    ("read",),
                ),
            ),
            behavior_proof_refs=("proof:bootstrap", "proof:daemon-health"),
        )
        declared = declare_durable_change_train(
            train_id=f"train:{tier.value}:open-v{slot}",
            tier=tier,
            current_version=slot - 1,
            target_version=slot,
            slot=slot,
            owner_ref="owner:open",
            migration=claim,
            riders=(rider,),
            backup_plan_ref=None if claim.requires_backup is False else "plan:daemon-open-verified-backup",
            declared_at_ms=1,
        )
        (tier_package / f"{slot:03d}.train.json").write_text(
            json.dumps(durable_change_train_to_payload(declared)), "utf-8"
        )
    monkeypatch.syspath_prepend(str(tmp_path))
    canonical_package = migration_runner._migration_package
    monkeypatch.setattr(
        migration_runner,
        "_migration_package",
        lambda observed: f"{package}.{tier.value}" if observed is tier else canonical_package(observed),
    )
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda observed: f"{package}.{tier.value}" if observed is tier else canonical_package(observed),
    )
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train.DURABLE_MIGRATION_ADOPTION_FLOORS",
        {ArchiveTier.SOURCE: 1, ArchiveTier.USER: 1},
    )
    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[tier] = 1 + len(steps)
    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)
    monkeypatch.setattr(bootstrap, "ARCHIVE_VERSION_BY_TIER", versions)
    monkeypatch.setattr("polylogue.storage.sqlite.archive_tiers.ARCHIVE_VERSION_BY_TIER", versions)
    monkeypatch.setattr("polylogue.operations.durable_change_train.ARCHIVE_VERSION_BY_TIER", versions)
    ddl = dict(ARCHIVE_DDL_BY_TIER)
    tables = "\n".join(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY) STRICT;" for _sql, table in steps)
    ddl[tier] = f"{base_ddl if base_ddl is not None else ddl[tier]}\n{tables}"
    monkeypatch.setattr(bootstrap, "ARCHIVE_DDL_BY_TIER", ddl)
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)


def _lease(root: Path) -> object:
    def lease(actor: str) -> AbstractContextManager[object]:
        return write_lease(actor, archive_root=root)

    return lease


def _apply(root: Path) -> tuple[object, ...]:
    with acquire_durable_archive_ownership(root, owner_id="daemon:test") as owner:
        return apply_declared_durable_migrations(root, archive_owner=owner, write_lease=_lease(root))  # type: ignore[arg-type]


def _backup_manifests(root: Path) -> set[Path]:
    """Pre-migration backups present now (fresh bootstrap may already own some)."""
    backups = root / ".maintenance-state" / "pre-migration-backups"
    return set(backups.rglob("manifest.json")) if backups.exists() else set()


def _version_and_table(path: Path, table: str) -> tuple[int, bool]:
    with sqlite3.connect(path) as conn:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        present = conn.execute("SELECT 1 FROM sqlite_schema WHERE name = ?", (table,)).fetchone() is not None
    return version, present


def test_an_archive_at_the_runtime_version_is_left_alone(cli_workspace: dict[str, Path]) -> None:
    root = cli_workspace["archive_root"]
    before = _backup_manifests(root)

    assert _apply(root) == ()
    # Applying nothing creates no backup of its own.
    assert _backup_manifests(root) == before


def test_an_additive_migration_applies_at_open_without_a_backup(
    one_shot_workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_BASELINE_DDL_BY_TIER

    root = one_shot_workspace_env["archive_root"]
    bootstrap_baseline_archive(root, monkeypatch)
    base = ARCHIVE_BASELINE_DDL_BY_TIER[ArchiveTier.SOURCE]
    _declare_future_migration(
        tmp_path,
        monkeypatch,
        ArchiveTier.SOURCE,
        sql="-- migration-safety: additive-no-backup\nCREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;\n",
        new_table="future_items",
        base_ddl=base,
    )
    source_db = root / "source.db"

    applied = _apply(root)

    assert [(item.tier, item.requires_backup) for item in applied] == [(ArchiveTier.SOURCE, False)]  # type: ignore[attr-defined]
    assert _version_and_table(source_db, "future_items") == (2, True)
    assert not (root / ".maintenance-state" / "pre-migration-backups").exists()
    assert _apply(root) == ()


def test_a_data_changing_migration_applies_only_behind_a_verified_backup(
    cli_workspace: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = cli_workspace["archive_root"]
    before = _backup_manifests(root)
    _declare_future_migration(
        tmp_path,
        monkeypatch,
        ArchiveTier.USER,
        sql="CREATE TABLE future_user_items (id INTEGER PRIMARY KEY) STRICT;\n",
        new_table="future_user_items",
    )

    applied = _apply(root)

    assert [(item.tier, item.requires_backup) for item in applied] == [(ArchiveTier.USER, True)]  # type: ignore[attr-defined]
    assert _version_and_table(root / "user.db", "future_user_items") == (2, True)
    (manifest,) = _backup_manifests(root) - before
    assert (manifest.parent / "verification-receipt.json").is_file()


def test_an_archive_two_steps_behind_rehearses_and_applies_each_step(
    cli_workspace: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full-chain rehearsal captures intermediate schemas before step backups."""

    root = cli_workspace["archive_root"]
    _declare_future_migrations(
        tmp_path,
        monkeypatch,
        ArchiveTier.USER,
        steps=(
            ("CREATE TABLE first_items (id INTEGER PRIMARY KEY) STRICT;\n", "first_items"),
            ("CREATE TABLE second_items (id INTEGER PRIMARY KEY) STRICT;\n", "second_items"),
        ),
    )

    applied = _apply(root)

    assert [item.target_version for item in applied] == [2, 3]  # type: ignore[attr-defined]
    assert _version_and_table(root / "user.db", "first_items") == (3, True)
    assert _version_and_table(root / "user.db", "second_items") == (3, True)

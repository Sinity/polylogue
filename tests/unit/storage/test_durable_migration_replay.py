"""A persisted migration witness records every schema-only step to canonical DDL."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.storage.sqlite import durable_change_train as train_module
from polylogue.storage.sqlite import migration_runner
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import DurableChangeTrainError, rehearse_durable_migration_chain


def _install_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    package = "fixture_replay_chain"
    source = tmp_path / package / "source"
    source.mkdir(parents=True)
    (tmp_path / package / "__init__.py").write_text("", encoding="utf-8")
    (source / "__init__.py").write_text("", encoding="utf-8")
    step_2 = "-- migration-safety: additive-no-backup\nCREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT NOT NULL) STRICT;\n"
    step_3 = "-- migration-safety: additive-no-backup\nCREATE INDEX items_label ON items(label);\n"
    (source / "002_items.sql").write_text(step_2, encoding="utf-8")
    (source / "003_items_label.sql").write_text(step_3, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(migration_runner, "_migration_package", lambda _tier: f"{package}.source")
    monkeypatch.setattr(train_module, "_migration_package", lambda _tier: f"{package}.source")
    monkeypatch.setattr(
        train_module,
        "validate_durable_migration_sidecars",
        lambda _tier, _migrations: (SimpleNamespace(slot=2), SimpleNamespace(slot=3)),
    )
    versions = dict(migration_runner.ARCHIVE_VERSION_BY_TIER)
    versions[ArchiveTier.SOURCE] = 3
    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)
    ddl = dict(migration_runner.ARCHIVE_DDL_BY_TIER)
    final_ddl = "CREATE TABLE base_items (id INTEGER PRIMARY KEY) STRICT;\n" + step_2 + step_3
    ddl[ArchiveTier.SOURCE] = final_ddl
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)
    return source, final_ddl


def test_full_numbered_chain_captures_intermediate_inventory_and_final_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, _final_ddl = _install_chain(tmp_path, monkeypatch)
    with closing(sqlite3.connect(":memory:")) as live:
        live.execute("CREATE TABLE base_items (id INTEGER PRIMARY KEY) STRICT")
        live.execute("PRAGMA user_version = 1")
        live.commit()
        proof = rehearse_durable_migration_chain(
            live,
            ArchiveTier.SOURCE,
            target_version=3,
            evidence_ref="proof:two-step-rehearsal",
        )

    assert proof.matches is True
    assert tuple((step.version, step.name) for step in proof.steps) == (
        (2, "002_items.sql"),
        (3, "003_items_label.sql"),
    )
    assert proof.steps[0].before_schema_inventory_sha256 == proof.original_schema_inventory_sha256
    assert proof.steps[0].after_schema_inventory_sha256 == proof.steps[1].before_schema_inventory_sha256
    assert proof.steps[-1].after_schema_inventory_sha256 == proof.terminal_schema_inventory_sha256
    assert proof.terminal_schema_inventory_sha256 == proof.canonical_schema_inventory_sha256

    disconnected_steps = (
        proof.steps[0],
        replace(proof.steps[1], before_schema_inventory_sha256="d" * 64),
    )
    disconnected = replace(proof, steps=disconnected_steps, chain_sha256="")
    disconnected = replace(
        disconnected,
        chain_sha256=migration_runner._migration_replay_chain_digest(
            tier=disconnected.tier,
            from_version=disconnected.from_version,
            target_version=disconnected.target_version,
            original_schema_inventory_sha256=disconnected.original_schema_inventory_sha256,
            steps=disconnected.steps,
            terminal_schema_inventory_sha256=disconnected.terminal_schema_inventory_sha256,
            canonical_version=disconnected.canonical_version,
            canonical_schema_inventory_sha256=disconnected.canonical_schema_inventory_sha256,
        ),
    )
    with pytest.raises(DurableChangeTrainError, match="adjacent step inventories do not join"):
        migration_runner.validate_durable_migration_replay_proof(disconnected)


def test_recovery_rejects_changed_installed_sql_for_a_persisted_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _final_ddl = _install_chain(tmp_path, monkeypatch)
    with closing(sqlite3.connect(":memory:")) as live:
        live.execute("CREATE TABLE base_items (id INTEGER PRIMARY KEY) STRICT")
        live.execute("PRAGMA user_version = 1")
        live.commit()
        proof = rehearse_durable_migration_chain(
            live,
            ArchiveTier.SOURCE,
            target_version=3,
            evidence_ref="proof:two-step-rehearsal",
        )

    (source / "002_items.sql").write_text(
        "-- migration-safety: additive-no-backup\nCREATE TABLE changed_items (id INTEGER PRIMARY KEY) STRICT;\n",
        encoding="utf-8",
    )
    with pytest.raises(DurableChangeTrainError, match="installed migration SQL and versions"):
        migration_runner.validate_durable_migration_replay_proof(proof, recompute_installed_bindings=True)


def test_a_released_historical_prefix_remains_valid_under_a_later_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _final_ddl = _install_chain(tmp_path, monkeypatch)
    original_versions = dict(migration_runner.ARCHIVE_VERSION_BY_TIER)
    original_ddl = dict(migration_runner.ARCHIVE_DDL_BY_TIER)
    versions = dict(original_versions)
    versions[ArchiveTier.SOURCE] = 2
    ddl = dict(original_ddl)
    ddl[ArchiveTier.SOURCE] = "CREATE TABLE base_items (id INTEGER PRIMARY KEY) STRICT;\n" + (
        source / "002_items.sql"
    ).read_text(encoding="utf-8")
    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)
    with closing(sqlite3.connect(":memory:")) as live:
        live.execute("CREATE TABLE base_items (id INTEGER PRIMARY KEY) STRICT")
        live.execute("PRAGMA user_version = 1")
        live.commit()
        historical = rehearse_durable_migration_chain(
            live,
            ArchiveTier.SOURCE,
            target_version=2,
            evidence_ref="proof:historical-v2-release",
        )

    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", original_versions)
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", original_ddl)
    migration_runner.validate_durable_migration_replay_proof(
        historical,
        recompute_installed_bindings=True,
    )

    (source / "002_items.sql").write_text(
        "-- migration-safety: additive-no-backup\nCREATE TABLE changed_items (id INTEGER PRIMARY KEY) STRICT;\n",
        encoding="utf-8",
    )
    with pytest.raises(DurableChangeTrainError, match="installed migration SQL and versions"):
        migration_runner.validate_durable_migration_replay_proof(
            historical,
            recompute_installed_bindings=True,
        )

"""Lifecycle contracts for durable source/user schema-change authority."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import polylogue.storage.sqlite.durable_change_train as durable_change_train_module
from polylogue.daemon.backup import backup_archive
from polylogue.operations.durable_change_train import (
    _audit_live_metadata,
    _write_immutable_audit_adoption_receipt,
    acquire_durable_archive_ownership,
    adopt_missing_audit_tier,
    audit_adoption_receipt_path,
    restore_adopted_audit_tier,
    validate_audit_adoption_receipt,
)
from polylogue.storage.sqlite import migration_runner
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import (
    DURABLE_MIGRATION_ADOPTION_FLOORS,
    _runtime_consumer_results,
    durable_change_train_manifest_path,
    durable_change_train_policy_report,
    execute_durable_change_train,
    reconcile_durable_change_train_startup,
    validate_durable_migration_sidecars,
)
from polylogue.storage.sqlite.migration_runner import (
    DurableChangeRider,
    DurableChangeTrain,
    DurableChangeTrainApplyError,
    DurableChangeTrainError,
    DurableChangeTrainRecoveryError,
    DurableChangeTrainState,
    DurableFailureClassification,
    DurableFreshDDLParityProof,
    DurableMigrationClaim,
    DurableRuntimeConsumer,
    DurableRuntimeConsumerResult,
    MigrationError,
    add_durable_change_train_rider,
    admit_durable_change_train,
    apply_durable_change_train,
    authorize_durable_change_train_backup,
    capture_durable_restart_convergence,
    declare_durable_change_train,
    durable_migration_claim_for_sql,
    durable_migration_collision_report,
    load_durable_change_train_manifest,
    prove_durable_change_train,
    prove_durable_fresh_ddl_parity,
    reconcile_interrupted_durable_change_train,
    record_durable_writer_release,
    recover_durable_change_train,
    release_durable_change_train,
    reserve_durable_change_train,
    write_durable_change_train_manifest,
)
from tests.infra.durable_schema_reset import reset_source_fixture_to_version
from tests.infra.durable_tier_fixtures import checkpoint_durable_tier, refresh_archive_format_marker

_CURRENT_VERSION = 1
_TARGET_VERSION = 2
_EMPTY_LIVENESS_DIGEST = hashlib.sha256(b"[]").hexdigest()
_BASE_ITEMS_DDL = "CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;"
_ADDITIVE_SQL = """-- migration-safety: additive-no-backup
CREATE TABLE durable_items (
    item_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
) STRICT;
"""


@contextmanager
def _memory_target(*, include_durable_items: bool = True) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT")
        if include_durable_items:
            conn.execute("CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT")
        conn.execute(f"PRAGMA user_version = {_TARGET_VERSION}")
        conn.commit()
        yield conn
    finally:
        conn.close()


def _create_current_database(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT")
        conn.execute("INSERT INTO base_items VALUES ('base-1', 'preserve-me')")
        conn.execute(f"PRAGMA user_version = {_CURRENT_VERSION}")
        conn.commit()


def test_explicit_migration_refuses_an_unmarked_historical_v1_before_writes(tmp_path: Path) -> None:
    """The production migration route needs the fresh-lineage marker, not v1 alone.

    Anti-vacuity: removing the format admission at the execution route lets this
    historical file reach migration reconciliation and changes its observable
    error from a lineage refusal to a guessed compatibility path.
    """
    source_path = tmp_path / "source.db"
    with sqlite3.connect(source_path) as conn:
        conn.execute("CREATE TABLE historical_v1 (id INTEGER PRIMARY KEY) STRICT")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    before = source_path.read_bytes()

    with pytest.raises(DurableChangeTrainError, match="archive format marker is missing"):
        execute_durable_change_train(
            tmp_path,
            ArchiveTier.SOURCE,
            backup_manifest=None,
            daemon_stopped_evidence_ref="proof:test-daemon-stopped",
            single_writer_evidence_ref="proof:test-single-writer",
            release_archive_ownership=lambda: pytest.fail("lineage refusal must precede writer release"),
        )

    assert source_path.read_bytes() == before


def _claim(tier: ArchiveTier, sql: str = _ADDITIVE_SQL) -> DurableMigrationClaim:
    return durable_migration_claim_for_sql(
        tier,
        "002_durable_items.sql",
        sql,
        owner_ref=f"owner:migration:{tier.value}:002",
    )


def _rider(*, consumer_count: int = 2, trust_floor_exception_ref: str | None = None) -> DurableChangeRider:
    consumers = tuple(
        DurableRuntimeConsumer(
            consumer_id=f"consumer-{index}",
            production_ref=f"polylogue/storage/{'writer' if index == 0 else 'reader'}_{index}.py:consume",
            behavior_proof_ref=f"proof:behavior:{index}",
            roles=("write" if index == 0 else "read",),
        )
        for index in range(consumer_count)
    )
    return DurableChangeRider(
        rider_id="rider:durable-items",
        owner_ref="owner:rider",
        schema_objects=("table:durable_items",),
        runtime_consumers=consumers,
        behavior_proof_refs=tuple(consumer.behavior_proof_ref for consumer in consumers),
        trust_floor_exception_ref=trust_floor_exception_ref,
    )


def _production_rider() -> DurableChangeRider:
    return DurableChangeRider(
        rider_id="rider:startup",
        owner_ref="owner:startup-rider",
        schema_objects=("table:durable_items",),
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


def _source_hook_event_production_rider() -> DurableChangeRider:
    return DurableChangeRider(
        rider_id="rider:source-hook-event",
        owner_ref="owner:source-hook-event",
        schema_objects=("table:raw_hook_events",),
        runtime_consumers=(
            DurableRuntimeConsumer(
                "source-hook-event-writer",
                "polylogue/storage/sqlite/archive_tiers/source_write.py:write_source_hook_event",
                "proof:source-v27:raw-hook-events-origin-repair",
                ("write",),
            ),
            DurableRuntimeConsumer(
                "source-hook-event-bootstrap",
                "polylogue/storage/sqlite/archive_tiers/bootstrap.py:initialize_archive_tier",
                "proof:source-v27:raw-hook-events-origin-repair",
                ("read",),
            ),
        ),
        behavior_proof_refs=("proof:source-v27:raw-hook-events-origin-repair",),
    )


def _parity(tier: ArchiveTier, *, include_durable_items: bool = True) -> DurableFreshDDLParityProof:
    with _memory_target(include_durable_items=include_durable_items) as migrated:
        with _memory_target() as fresh:
            return prove_durable_fresh_ddl_parity(
                tier,
                _TARGET_VERSION,
                migrated_connection=migrated,
                fresh_connection=fresh,
                evidence_ref=f"proof:fresh-ddl:{tier.value}",
            )


@contextmanager
def _ddl_target(ddl: str) -> Iterator[sqlite3.Connection]:
    """Build one in-memory tier at ``_TARGET_VERSION`` from an explicit DDL script."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(ddl)
        conn.execute(f"PRAGMA user_version = {_TARGET_VERSION}")
        conn.commit()
        yield conn
    finally:
        conn.close()


def _parity_for_ddl(tier: ArchiveTier, ddl: str) -> DurableFreshDDLParityProof:
    """Parity proof for a fixture whose canonical shape is not the toy tier."""
    with _ddl_target(ddl) as migrated, _ddl_target(ddl) as fresh:
        return prove_durable_fresh_ddl_parity(
            tier,
            _TARGET_VERSION,
            migrated_connection=migrated,
            fresh_connection=fresh,
            evidence_ref=f"proof:fresh-ddl:{tier.value}",
        )


def _declared(
    tier: ArchiveTier,
    *,
    claim: DurableMigrationClaim | None = None,
    rider: DurableChangeRider | None = None,
    owner_ref: str = "owner:train",
    backup_plan_ref: str | None = None,
) -> DurableChangeTrain:
    migration = claim or _claim(tier)
    return declare_durable_change_train(
        train_id=f"train:{tier.value}:v{_TARGET_VERSION}",
        tier=tier,
        current_version=_CURRENT_VERSION,
        target_version=_TARGET_VERSION,
        slot=_TARGET_VERSION,
        owner_ref=owner_ref,
        migration=migration,
        riders=((rider or _rider()),),
        backup_plan_ref=backup_plan_ref,
        declared_at_ms=1,
    )


def _admitted(
    tier: ArchiveTier,
    *,
    claim: DurableMigrationClaim | None = None,
    rider: DurableChangeRider | None = None,
    owner_ref: str = "owner:train",
    backup_plan_ref: str | None = None,
    active_trains: tuple[DurableChangeTrain, ...] = (),
    parity: DurableFreshDDLParityProof | None = None,
) -> DurableChangeTrain:
    migration = claim or _claim(tier)
    return admit_durable_change_train(
        _declared(
            tier,
            claim=migration,
            rider=rider,
            owner_ref=owner_ref,
            backup_plan_ref=backup_plan_ref,
        ),
        observed_current_version=_CURRENT_VERSION,
        fresh_ddl_parity=parity if parity is not None else _parity(tier),
        admission_evidence_ref=f"proof:admit:{tier.value}",
        active_trains=active_trains,
        migration_claims=(migration,),
        canonical_target_version=_TARGET_VERSION,
        admitted_at_ms=2,
    )


#: The synthetic fixtures below own slot ``002`` -- the first slot any durable
#: tier of this lineage may own, because ``ARCHIVE_FORMAT_FLOOR_VERSION`` is 1.
_SYNTHETIC_SIDECAR_NAME = f"{_TARGET_VERSION:03d}.train.json"

_SOURCE_ADOPTION_FLOOR = DURABLE_MIGRATION_ADOPTION_FLOORS[ArchiveTier.SOURCE]
# The first slot a source train may own. Synthetic future-migration fixtures
# must sit above the floor, or sidecar discovery refuses them before the
# behavior under test runs.
_NEXT_SOURCE_SLOT = _SOURCE_ADOPTION_FLOOR + 1
_NEXT_SOURCE_SQL_NAME = f"{_NEXT_SOURCE_SLOT:03d}_future_items.sql"
_NEXT_SOURCE_SIDECAR_NAME = f"{_NEXT_SOURCE_SLOT:03d}.train.json"


#: Every module that rebinds the durable version map at import time, plus the
#: package attribute the lazily-importing durable-train helpers read. A test
#: that pins only one of them leaves the others disagreeing about the tier's
#: target, which no production configuration ever does.
_VERSION_MAP_OWNERS = (
    "polylogue.storage.sqlite.archive_tiers.ARCHIVE_VERSION_BY_TIER",
    "polylogue.storage.sqlite.archive_tiers.bootstrap.ARCHIVE_VERSION_BY_TIER",
    "polylogue.storage.sqlite.archive_tiers.archive_plan.ARCHIVE_VERSION_BY_TIER",
    "polylogue.storage.sqlite.migration_runner.ARCHIVE_VERSION_BY_TIER",
    "polylogue.operations.durable_change_train.ARCHIVE_VERSION_BY_TIER",
)


def _pin_source_runtime_version(monkeypatch: pytest.MonkeyPatch, version: int) -> dict[ArchiveTier, int]:
    """Pin the source tier's durable target everywhere the runtime reads it."""
    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[ArchiveTier.SOURCE] = version
    for target in _VERSION_MAP_OWNERS:
        monkeypatch.setattr(target, versions)
    return versions


def _install_synthetic_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier: ArchiveTier,
    *,
    sql: str = _ADDITIVE_SQL,
    canonical_base: str = _BASE_ITEMS_DDL,
) -> str:
    """Install one synthetic slot-002 migration *with* its checked-in sidecar.

    Returns the canonical DDL it declared for the tier, so a caller that needs
    a parity proof over the same shape does not restate the composition.

    ``002`` sits above ``ARCHIVE_FORMAT_FLOOR_VERSION``, and production
    discovery (``validate_durable_migration_sidecars``) refuses any post-floor
    SQL slot that has no frozen ``NNN.train.json`` beside it. A fixture that
    shipped the SQL alone therefore never reached the behaviour under test; it
    tripped the sidecar requirement first. Writing the sidecar here reproduces
    what the repository itself must carry for slot 002.
    """
    package_name = f"fixture_migrations_{tier.value}_{tmp_path.name.replace('-', '_')}"
    package_root = tmp_path / package_name
    tier_package = package_root / tier.value
    tier_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (tier_package / "__init__.py").write_text("", encoding="utf-8")
    (tier_package / "002_durable_items.sql").write_text(sql, encoding="utf-8")
    declared = declare_durable_change_train(
        train_id=f"train:{tier.value}:v{_TARGET_VERSION}",
        tier=tier,
        current_version=_CURRENT_VERSION,
        target_version=_TARGET_VERSION,
        slot=_TARGET_VERSION,
        owner_ref=f"owner:migration:{tier.value}:002",
        migration=_claim(tier, sql),
        riders=(_rider(),),
        declared_at_ms=1,
    )
    (tier_package / _SYNTHETIC_SIDECAR_NAME).write_text(
        json.dumps(migration_runner.durable_change_train_to_payload(declared)), encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[tier] = _TARGET_VERSION
    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)
    # Once a slot carries a sidecar, the runner additionally proves the
    # migrated tier against ``ARCHIVE_DDL_BY_TIER[tier]`` before it commits.
    # The synthetic tier's canonical shape is the current database plus what
    # this slot adds, so declare exactly that; leaving the real source/user
    # DDL in place would compare a two-table fixture against the whole
    # shipped schema and fail on every object the fixture never had. A test
    # whose behaviour needs the real tier -- a production runtime-consumer
    # probe, say -- passes that tier's shipped DDL as ``canonical_base`` and
    # bootstraps the archive through the production route instead.
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    ddl = dict(ARCHIVE_DDL_BY_TIER)
    canonical = f"{canonical_base}\n{sql}"
    ddl[tier] = canonical
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)
    monkeypatch.setattr(bootstrap, "ARCHIVE_DDL_BY_TIER", ddl)
    monkeypatch.setattr(
        migration_runner, "_migration_package", lambda observed_tier: f"{package_name}.{observed_tier.value}"
    )
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda observed_tier: f"{package_name}.{observed_tier.value}",
    )
    return canonical


def _reserve_and_authorize(
    conn: sqlite3.Connection,
    train: DurableChangeTrain,
    *,
    archive_root: Path,
) -> DurableChangeTrain:
    reserved = reserve_durable_change_train(
        train,
        reservation_id="lease:archive-root",
        reservation_owner_ref=train.owner_ref,
        archive_root=archive_root,
        tier_path=archive_root / f"{train.tier.value}.db",
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:rebuild-lease",
        reserved_at_ms=3,
    )
    return authorize_durable_change_train_backup(
        conn,
        reserved,
        backup_manifest=None,
        evidence_ref="proof:additive-no-backup",
        authorized_at_ms=4,
    )


def _runtime_results() -> tuple[DurableRuntimeConsumerResult, ...]:
    return (
        DurableRuntimeConsumerResult("consumer-0", "proof:behavior:0", True),
        DurableRuntimeConsumerResult("consumer-1", "proof:behavior:1", True),
    )


def test_applied_train_release_requires_the_source_hook_event_writer_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release needs the shipped hook-event writer to actually write.

    The rider names ``source_write.py:write_source_hook_event`` as a runtime
    consumer, and the probe adapter calls it for real. That needs the source
    tier's own shipped schema -- ``raw_hook_events`` in particular -- so this
    fixture bootstraps a real archive at the adoption floor and layers the
    synthetic slot on top of the canonical DDL rather than on the two-table
    toy the other lifecycle tests use.

    Anti-vacuity: drop ``source-hook-event-writer`` from the rider's runtime
    consumers and ``released.proof.runtime_consumers`` no longer carries it,
    so the ``next(...)`` below raises ``StopIteration``.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    _pin_source_runtime_version(monkeypatch, _SOURCE_ADOPTION_FLOOR)
    initialize_active_archive_root(tmp_path)
    db_path = tmp_path / "source.db"
    canonical_ddl = _install_synthetic_migration(
        tmp_path,
        monkeypatch,
        ArchiveTier.SOURCE,
        canonical_base=ARCHIVE_DDL_BY_TIER[ArchiveTier.SOURCE],
    )
    train = _admitted(
        ArchiveTier.SOURCE,
        rider=_source_hook_event_production_rider(),
        parity=_parity_for_ddl(ArchiveTier.SOURCE, canonical_ddl),
    )
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        train = apply_durable_change_train(conn, train)

    train = record_durable_writer_release(train, evidence_ref="proof:source-hook-event-writer-release")
    with sqlite3.connect(db_path) as restarted:
        actual_parity = _parity_for_ddl(ArchiveTier.SOURCE, canonical_ddl)
        runtime_results = _runtime_consumer_results(train, tmp_path)
        restart = capture_durable_restart_convergence(
            restarted,
            train,
            runtime_consumers=runtime_results,
            evidence_ref="proof:source-hook-event-restart",
        )
    train = prove_durable_change_train(
        train,
        fresh_ddl_parity=actual_parity,
        runtime_consumers=runtime_results,
        restart_convergence=restart,
    )
    released = release_durable_change_train(train, evidence_ref="proof:source-hook-event-release")

    assert released.state is DurableChangeTrainState.RELEASED
    assert released.proof is not None
    assert released.apply_evidence is not None
    hook_writer = next(
        result for result in released.proof.runtime_consumers if result.consumer_id == "source-hook-event-writer"
    )
    assert hook_writer.passed is True


@pytest.mark.parametrize("tier", (ArchiveTier.SOURCE, ArchiveTier.USER))
def test_synthetic_source_and_user_trains_complete_the_full_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier: ArchiveTier,
) -> None:
    """Both durable tiers persist every state and use the shipped migration transaction."""
    db_path = tmp_path / f"{tier.value}.db"
    manifest = tmp_path / f"{tier.value}-train.json"
    _create_current_database(db_path)
    _install_synthetic_migration(tmp_path, monkeypatch, tier)

    def persist_and_reload(next_train: DurableChangeTrain, expected_revision: int) -> DurableChangeTrain:
        write_durable_change_train_manifest(
            manifest,
            next_train,
            expected_revision=expected_revision,
        )
        return load_durable_change_train_manifest(manifest)

    claim = _claim(tier)
    train = _declared(tier, claim=claim)
    train = persist_and_reload(train, -1)
    previous_revision = train.revision
    train = admit_durable_change_train(
        train,
        observed_current_version=_CURRENT_VERSION,
        fresh_ddl_parity=_parity(tier),
        admission_evidence_ref=f"proof:admit:{tier.value}",
        migration_claims=(claim,),
        canonical_target_version=_TARGET_VERSION,
        admitted_at_ms=2,
    )
    train = persist_and_reload(train, previous_revision)
    previous_revision = train.revision
    train = reserve_durable_change_train(
        train,
        reservation_id="lease:archive-root",
        reservation_owner_ref=train.owner_ref,
        archive_root=tmp_path,
        tier_path=db_path,
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:rebuild-lease",
    )
    train = persist_and_reload(train, previous_revision)
    previous_revision = train.revision
    with sqlite3.connect(db_path) as conn:
        train = authorize_durable_change_train_backup(
            conn,
            train,
            backup_manifest=None,
            evidence_ref="proof:additive-no-backup",
        )
        train = persist_and_reload(train, previous_revision)
        previous_revision = train.revision
        train = apply_durable_change_train(conn, train)
        train = persist_and_reload(train, previous_revision)
        assert conn.execute("SELECT payload FROM base_items WHERE item_id='base-1'").fetchone() == ("preserve-me",)
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='durable_items'").fetchone() == (
            "durable_items",
        )
    assert train.apply_evidence is not None
    assert train.apply_evidence.migration_result.applied_versions == (_TARGET_VERSION,)
    assert train.apply_evidence.row_parity.ok is True

    previous_revision = train.revision
    train = record_durable_writer_release(train, evidence_ref="proof:lease-released")
    train = persist_and_reload(train, previous_revision)
    with sqlite3.connect(db_path) as restarted:
        with _memory_target() as fresh:
            actual_parity = prove_durable_fresh_ddl_parity(
                tier,
                _TARGET_VERSION,
                migrated_connection=restarted,
                fresh_connection=fresh,
                evidence_ref=f"proof:post-apply-fresh:{tier.value}",
            )
        runtime_results = _runtime_results()
        restart = capture_durable_restart_convergence(
            restarted,
            train,
            runtime_consumers=runtime_results,
            evidence_ref="proof:runtime-restart",
        )
    previous_revision = train.revision
    train = prove_durable_change_train(
        train,
        fresh_ddl_parity=actual_parity,
        runtime_consumers=runtime_results,
        restart_convergence=restart,
    )
    train = persist_and_reload(train, previous_revision)
    previous_revision = train.revision
    train = release_durable_change_train(train, evidence_ref="proof:train-release")
    train = persist_and_reload(train, previous_revision)

    assert train.state is DurableChangeTrainState.RELEASED
    assert train.revision == 7


def test_future_train_sidecar_discovery_uses_real_package_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "fixture_migrations"
    source_package = package_root / "source"
    source_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (source_package / "__init__.py").write_text("", encoding="utf-8")
    sql = "-- migration-safety: additive-no-backup\nCREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;\n"
    sql_path = source_package / _NEXT_SOURCE_SQL_NAME
    sql_path.write_text(sql, encoding="utf-8")
    claim = durable_migration_claim_for_sql(
        ArchiveTier.SOURCE,
        sql_path.name,
        sql,
        owner_ref="owner:future-source",
    )
    train = declare_durable_change_train(
        train_id=f"train:source:v{_NEXT_SOURCE_SLOT}",
        tier=ArchiveTier.SOURCE,
        current_version=_SOURCE_ADOPTION_FLOOR,
        target_version=_NEXT_SOURCE_SLOT,
        slot=_NEXT_SOURCE_SLOT,
        owner_ref="owner:future-source",
        migration=claim,
        riders=(_rider(),),
        declared_at_ms=1,
    )
    (source_package / _NEXT_SOURCE_SIDECAR_NAME).write_text(
        json.dumps(migration_runner.durable_change_train_to_payload(train)),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda _tier: "fixture_migrations.source",
    )
    monkeypatch.setattr(
        migration_runner,
        "_migration_package",
        lambda _tier: "fixture_migrations.source",
    )

    observed = validate_durable_migration_sidecars(ArchiveTier.SOURCE, ((sql_path.name, sql),))
    loaded = migration_runner._load_migrations(ArchiveTier.SOURCE)

    assert [item.resource_name for item in observed] == [_NEXT_SOURCE_SIDECAR_NAME]
    assert observed[0].train.migration.sql_sha256 == claim.sql_sha256
    assert loaded[0].version == _NEXT_SOURCE_SLOT
    assert "fixture_migrations.source" in sys.modules

    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[ArchiveTier.SOURCE] = _NEXT_SOURCE_SLOT
    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)
    ddl = dict(ARCHIVE_DDL_BY_TIER)
    ddl[ArchiveTier.SOURCE] = """
    CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;
    CREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;
    """
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)
    db_path = tmp_path / "real-route-source.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT")
        conn.execute(f"PRAGMA user_version = {_SOURCE_ADOPTION_FLOOR}")
        conn.commit()
        result = migration_runner.migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=None)
        assert result.applied_versions == (_NEXT_SOURCE_SLOT,)
        assert conn.execute("PRAGMA user_version").fetchone() == (_NEXT_SOURCE_SLOT,)
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='future_items'").fetchone() == ("future_items",)


def test_maintenance_route_persists_and_proves_a_future_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An archive born at the floor advances through the slot-2 route.

    The archive is bootstrapped with the source target pinned to the adoption
    floor, so its format marker records a birth version of 1 -- the shape of
    every archive created before the first numbered source slot shipped. The
    runtime target is then raised to 2 and the train migrates it. Bootstrapping
    at the current target instead would model no migration at all once source
    itself sits above the floor.

    Anti-vacuity: removing the fresh floor from durable train discovery leaves
    slot 2 occupied by the retired history; removing marker admission lets an
    unmarked v1 tier enter this production route.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    _pin_source_runtime_version(monkeypatch, _SOURCE_ADOPTION_FLOOR)
    initialize_active_archive_root(tmp_path)
    package_root = tmp_path / "fixture_migrations_maintenance"
    source_package = package_root / "source"
    source_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (source_package / "__init__.py").write_text("", encoding="utf-8")
    sql = "-- migration-safety: additive-no-backup\nCREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;\n"
    (source_package / "002_future_items.sql").write_text(sql, encoding="utf-8")
    claim = durable_migration_claim_for_sql(
        ArchiveTier.SOURCE,
        "002_future_items.sql",
        sql,
        owner_ref="owner:maintenance-source",
    )
    rider = DurableChangeRider(
        rider_id="rider:maintenance",
        owner_ref="owner:maintenance-rider",
        schema_objects=("table:future_items",),
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
        train_id="train:source:v2",
        tier=ArchiveTier.SOURCE,
        current_version=1,
        target_version=2,
        slot=2,
        owner_ref="owner:maintenance-source",
        migration=claim,
        riders=(rider,),
        declared_at_ms=1,
    )
    (source_package / "002.train.json").write_text(
        json.dumps(migration_runner.durable_change_train_to_payload(declared)), encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(migration_runner, "_migration_package", lambda _tier: "fixture_migrations_maintenance.source")
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda _tier: "fixture_migrations_maintenance.source",
    )
    _pin_source_runtime_version(monkeypatch, 2)
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    ddl = dict(ARCHIVE_DDL_BY_TIER)
    ddl[ArchiveTier.SOURCE] = ARCHIVE_DDL_BY_TIER[ArchiveTier.SOURCE] + "\n" + sql
    monkeypatch.setattr(bootstrap, "ARCHIVE_DDL_BY_TIER", ddl)
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)
    db_path = tmp_path / "source.db"

    released: list[bool] = []
    result = execute_durable_change_train(
        tmp_path,
        ArchiveTier.SOURCE,
        backup_manifest=None,
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
        release_archive_ownership=lambda: released.append(True),
    )

    assert result.train is not None
    assert result.train.state is DurableChangeTrainState.RELEASED
    assert result.migration_result is not None
    assert result.migration_result.applied_versions == (2,)
    assert released == [True]
    manifest_path = durable_change_train_manifest_path(tmp_path, ArchiveTier.SOURCE, 2)
    assert load_durable_change_train_manifest(manifest_path).state is DurableChangeTrainState.RELEASED

    released_bytes = db_path.read_bytes()
    db_path.unlink()
    with pytest.raises(DurableChangeTrainError, match="missing durable tier"):
        execute_durable_change_train(
            tmp_path,
            ArchiveTier.SOURCE,
            backup_manifest=None,
            daemon_stopped_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
            release_archive_ownership=lambda: pytest.fail("missing released tier was released again"),
        )
    assert not db_path.exists()
    db_path.write_bytes(released_bytes)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    with pytest.raises(DurableChangeTrainError, match="historical version-1 schema"):
        execute_durable_change_train(
            tmp_path,
            ArchiveTier.SOURCE,
            backup_manifest=None,
            daemon_stopped_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
            release_archive_ownership=lambda: pytest.fail("stale released train was released again"),
        )


def test_maintenance_route_replays_historical_sidecars_before_current_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later shipped slot must not reject an earlier persisted train.

    The archive is bootstrapped at the adoption floor through the production
    route, so it carries its own format marker and bootstrap receipt, and the
    runtime target is only then raised to slot 3. A hand-authored source.db
    cannot stand in: ``execute_durable_change_train`` admits an archive by its
    ``.polylogue-format.json`` lineage marker first, which a bare fixture file
    does not have.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    _pin_source_runtime_version(monkeypatch, _SOURCE_ADOPTION_FLOOR)
    initialize_active_archive_root(tmp_path)

    package_root = tmp_path / "fixture_migrations_sequential"
    source_package = package_root / "source"
    source_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (source_package / "__init__.py").write_text("", encoding="utf-8")

    rider_template = DurableChangeRider(
        rider_id="rider:maintenance",
        owner_ref="owner:maintenance-rider",
        schema_objects=(),
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
    migrations = (
        (2, "durable_items", "CREATE TABLE durable_items (id INTEGER PRIMARY KEY) STRICT;"),
        (3, "later_items", "CREATE TABLE later_items (id INTEGER PRIMARY KEY) STRICT;"),
    )
    for slot, table_name, statement in migrations:
        sql = f"-- migration-safety: additive-no-backup\n{statement}\n"
        filename = f"{slot:03d}_{table_name}.sql"
        (source_package / filename).write_text(sql, encoding="utf-8")
        claim = durable_migration_claim_for_sql(
            ArchiveTier.SOURCE,
            filename,
            sql,
            owner_ref=f"owner:maintenance-source:{slot}",
        )
        rider = replace(
            rider_template,
            rider_id=f"rider:maintenance:{slot}",
            schema_objects=(f"table:{table_name}",),
        )
        declared = declare_durable_change_train(
            train_id=f"train:source:v{slot}",
            tier=ArchiveTier.SOURCE,
            current_version=slot - 1,
            target_version=slot,
            slot=slot,
            owner_ref=f"owner:maintenance-source:{slot}",
            migration=claim,
            riders=(rider,),
            declared_at_ms=slot,
        )
        (source_package / f"{slot:03d}.train.json").write_text(
            json.dumps(migration_runner.durable_change_train_to_payload(declared)), encoding="utf-8"
        )

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(migration_runner, "_migration_package", lambda _tier: "fixture_migrations_sequential.source")
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda _tier: "fixture_migrations_sequential.source",
    )
    _pin_source_runtime_version(monkeypatch, 3)
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    ddl = dict(ARCHIVE_DDL_BY_TIER)
    ddl[ArchiveTier.SOURCE] = "\n".join(
        (ARCHIVE_DDL_BY_TIER[ArchiveTier.SOURCE], *(statement for _slot, _table, statement in migrations))
    )
    monkeypatch.setattr(bootstrap, "ARCHIVE_DDL_BY_TIER", ddl)
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)
    db_path = tmp_path / "source.db"
    released: list[bool] = []

    first = execute_durable_change_train(
        tmp_path,
        ArchiveTier.SOURCE,
        backup_manifest=None,
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
        release_archive_ownership=lambda: released.append(True),
    )
    assert first.migration_result is not None
    assert first.migration_result.applied_versions == (2,)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone() == (2,)

    second = execute_durable_change_train(
        tmp_path,
        ArchiveTier.SOURCE,
        backup_manifest=None,
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
        release_archive_ownership=lambda: released.append(True),
    )
    assert second.migration_result is not None
    assert second.migration_result.applied_versions == (3,)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone() == (3,)
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='later_items'").fetchone() == ("later_items",)
    assert released == [True, True]
    historical_manifest = durable_change_train_manifest_path(tmp_path, ArchiveTier.SOURCE, 2)
    manifest_v3 = durable_change_train_manifest_path(tmp_path, ArchiveTier.SOURCE, 3)
    manifest_v3_bytes = manifest_v3.read_bytes()
    manifest_v3.unlink()
    with pytest.raises(DurableChangeTrainError, match=r"versions \[3\]"):
        durable_change_train_module.reconcile_durable_change_train_startup(tmp_path)
    assert released == [True, True]
    with pytest.raises(DurableChangeTrainError, match="lacks released train evidence"):
        execute_durable_change_train(
            tmp_path,
            ArchiveTier.SOURCE,
            backup_manifest=None,
            daemon_stopped_evidence_ref="proof:daemon-stopped",
            single_writer_evidence_ref="proof:archive-ownership-lock",
            release_archive_ownership=lambda: pytest.fail("missing intervening train was admitted"),
        )
    manifest_v3.write_bytes(manifest_v3_bytes)
    evidence_captures = 0
    source_canonical_targets: list[int] = []
    real_capture = migration_runner.capture_durable_database_evidence
    real_canonical_inventory = durable_change_train_module._canonical_schema_inventory

    def count_evidence_captures(
        connection: sqlite3.Connection, tier: ArchiveTier
    ) -> migration_runner.DurableDatabaseEvidence:
        nonlocal evidence_captures
        evidence_captures += 1
        return real_capture(connection, tier)

    def count_canonical_inventories(tier: ArchiveTier, target_version: int) -> migration_runner.DurableSchemaInventory:
        if tier is ArchiveTier.SOURCE:
            source_canonical_targets.append(target_version)
        return real_canonical_inventory(tier, target_version)

    monkeypatch.setattr(durable_change_train_module, "capture_durable_database_evidence", count_evidence_captures)
    monkeypatch.setattr(durable_change_train_module, "_canonical_schema_inventory", count_canonical_inventories)
    third = execute_durable_change_train(
        tmp_path,
        ArchiveTier.SOURCE,
        backup_manifest=None,
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
        release_archive_ownership=lambda: released.append(True),
    )
    assert third.forward_version_receipt is not None
    assert third.forward_version_receipt.historical_target_version == 2
    assert third.forward_version_receipt.observed_live_version == 3
    # The no-op pass reuses what it derived once: one live evidence capture
    # for the source tier, and one canonical inventory, for the live target it
    # observed. Re-deriving either per persisted manifest -- there are two,
    # v2 and v3 -- would show up here immediately.
    #
    # Only the source tier is counted. The other durable tiers contribute
    # their own corroboration inventories through
    # ``_fresh_durable_bootstrap_tier_is_own``, which is a different mechanism
    # with a different (currently repeated) call pattern; pinning a global
    # total here would make this assertion about that instead.
    assert evidence_captures == 1
    assert source_canonical_targets == [3]

    historical_train = load_durable_change_train_manifest(historical_manifest)
    with sqlite3.connect(db_path) as conn:
        actual = migration_runner.capture_durable_database_evidence(conn, ArchiveTier.SOURCE)
        receipt = durable_change_train_module._verify_released_train_live_tier(
            tmp_path,
            conn,
            historical_train,
            current_target_version=3,
            actual_evidence=actual,
        )
    assert receipt is not None
    assert receipt.historical_target_version == 2
    assert receipt.current_target_version == 3
    assert receipt.observed_live_version == 3
    assert historical_train.proof is not None
    assert (
        receipt.historical_schema_inventory_sha256 == historical_train.proof.fresh_ddl_parity.migrated_inventory_sha256
    )
    with sqlite3.connect(db_path) as conn:
        with pytest.raises(DurableChangeTrainError, match="is newer than current target"):
            durable_change_train_module._verify_released_train_live_tier(
                tmp_path,
                conn,
                historical_train,
                current_target_version=2,
                actual_evidence=actual,
            )

    captures = 0
    real_capture = migration_runner.capture_durable_database_evidence

    def count_captures(connection: sqlite3.Connection, tier: ArchiveTier) -> migration_runner.DurableDatabaseEvidence:
        nonlocal captures
        captures += 1
        return real_capture(connection, tier)

    monkeypatch.setattr(durable_change_train_module, "capture_durable_database_evidence", count_captures)
    assert reconcile_durable_change_train_startup(tmp_path) == (
        durable_change_train_manifest_path(tmp_path, ArchiveTier.SOURCE, 2),
        durable_change_train_manifest_path(tmp_path, ArchiveTier.SOURCE, 3),
    )
    assert captures == 1

    unrelated_root = tmp_path / "unrelated-archive"
    unrelated_root.mkdir()
    # The bootstrapped tier runs in WAL mode, so the committed slot-3 state
    # can still be sitting in ``source.db-wal``. Copying the main file alone
    # would hand the unrelated root a v2 image and make the refusal below a
    # fixture artifact instead of the identity check under test.
    with closing(sqlite3.connect(db_path)) as live:
        live.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shutil.copy2(db_path, unrelated_root / "source.db")
    unrelated_manifest = durable_change_train_manifest_path(unrelated_root, ArchiveTier.SOURCE, 2)
    unrelated_manifest.parent.mkdir(parents=True)
    shutil.copy2(historical_manifest, unrelated_manifest)
    shutil.copy2(manifest_v3, durable_change_train_manifest_path(unrelated_root, ArchiveTier.SOURCE, 3))
    with sqlite3.connect(unrelated_root / "source.db") as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        # Without the live slot in the copy the refusal below would fire for
        # a version mismatch and prove nothing about archive identity.
        assert conn.execute("PRAGMA user_version").fetchone() == (3,)
    with pytest.raises(DurableChangeTrainError, match="immutable archive identity differs"):
        reconcile_durable_change_train_startup(unrelated_root)

    # Remove the object the newest slot introduced: the live tier still says
    # v3 but no longer has the canonical v3 shape.
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"DROP TABLE {migrations[-1][1]}")
        conn.commit()
        tampered = migration_runner.capture_durable_database_evidence(conn, ArchiveTier.SOURCE)
        with pytest.raises(DurableChangeTrainError, match="canonical live version"):
            durable_change_train_module._verify_released_train_live_tier(
                tmp_path,
                conn,
                historical_train,
                current_target_version=3,
                actual_evidence=tampered,
            )


def test_released_train_chain_is_anchored_at_adoption_floor() -> None:
    floor = DURABLE_MIGRATION_ADOPTION_FLOORS[ArchiveTier.SOURCE]
    released = cast(DurableChangeTrain, SimpleNamespace(state=DurableChangeTrainState.RELEASED))

    with pytest.raises(DurableChangeTrainError, match=rf"versions \[{floor + 1}\]"):
        durable_change_train_module._require_released_train_chain(
            ArchiveTier.SOURCE,
            {
                floor + 2: released,
                floor + 3: released,
            },
            current_version=floor + 3,
        )


def test_released_train_chain_can_start_at_bootstrap_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    floor = DURABLE_MIGRATION_ADOPTION_FLOORS[ArchiveTier.SOURCE]
    current_version = floor + 1
    released = cast(DurableChangeTrain, SimpleNamespace(state=DurableChangeTrainState.RELEASED))
    monkeypatch.setattr(durable_change_train_module, "_historical_schema_evidence", lambda _train: None)

    durable_change_train_module._require_released_train_chain(
        ArchiveTier.SOURCE,
        {current_version: released},
        current_version=current_version,
        floor=floor,
    )


def test_forward_receipt_checks_missing_chain_before_empty_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    floor = DURABLE_MIGRATION_ADOPTION_FLOORS[ArchiveTier.SOURCE]
    current_version = floor + 2
    current_train = cast(
        DurableChangeTrain,
        SimpleNamespace(target_version=current_version, state=DurableChangeTrainState.RELEASED),
    )
    monkeypatch.setattr(
        durable_change_train_module,
        "_released_train_manifests_by_target",
        lambda _manifest_root, _tier: {current_version: current_train},
    )

    with (
        sqlite3.connect(":memory:") as conn,
        pytest.raises(
            DurableChangeTrainError,
            match=rf"versions \[{floor + 1}\]",
        ),
    ):
        durable_change_train_module._forward_version_receipt_for_current_tier(
            tmp_path,
            conn,
            ArchiveTier.SOURCE,
            current_version=current_version,
            current_target_version=current_version,
        )


def test_forward_receipt_skips_non_train_audit_tier(tmp_path: Path) -> None:
    with sqlite3.connect(":memory:") as conn:
        assert (
            durable_change_train_module._forward_version_receipt_for_current_tier(
                tmp_path,
                conn,
                ArchiveTier.AUDIT,
                current_version=1,
                current_target_version=1,
            )
            is None
        )


def test_startup_recovers_later_train_before_released_chain_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The first two slots this lineage can own. The pre-reset version of this
    # test named 27 and 28, which the adoption floor of 1 turns into a
    # twenty-five-slot gap with no released train evidence -- the chain check
    # then refuses before the ordering under test is ever observed.
    first_slot = _NEXT_SOURCE_SLOT
    later_slot = _NEXT_SOURCE_SLOT + 1
    manifest_root = tmp_path / ".maintenance-state" / "durable-change-trains"
    manifest_root.mkdir(parents=True)
    first_path = manifest_root / f"source-{first_slot:03d}.json"
    later_path = manifest_root / f"source-{later_slot:03d}.json"
    first_path.touch()
    later_path.touch()
    released = cast(
        DurableChangeTrain,
        SimpleNamespace(state=DurableChangeTrainState.RELEASED, tier=ArchiveTier.SOURCE, target_version=first_slot),
    )
    later_released = cast(
        DurableChangeTrain,
        SimpleNamespace(state=DurableChangeTrainState.RELEASED, tier=ArchiveTier.SOURCE, target_version=later_slot),
    )
    backup_authorized = cast(
        DurableChangeTrain,
        SimpleNamespace(
            state=DurableChangeTrainState.BACKUP_AUTHORIZED,
            tier=ArchiveTier.SOURCE,
            target_version=later_slot,
            train_id=f"train:source:v{later_slot}",
            revision=0,
        ),
    )
    states = {first_path: released, later_path: backup_authorized}
    events: list[tuple[str, int]] = []

    @contextmanager
    def fake_open_tier(_path: Path) -> Iterator[sqlite3.Connection]:
        with sqlite3.connect(":memory:") as connection:
            yield connection

    def fake_load(path: Path) -> DurableChangeTrain:
        return states[path]

    def fake_persist(path: Path, train: DurableChangeTrain, *, expected_revision: int) -> DurableChangeTrain:
        states[path] = train
        return train

    def fake_recover(*_args: object, **_kwargs: object) -> DurableChangeTrain:
        events.append(("recover", later_slot))
        return later_released

    def fake_capture(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(user_version=later_slot)

    monkeypatch.setattr(durable_change_train_module, "_open_existing_tier", fake_open_tier)
    monkeypatch.setattr(durable_change_train_module, "load_durable_change_train_manifest", fake_load)
    monkeypatch.setattr(durable_change_train_module, "_persist_train_transition", fake_persist)
    monkeypatch.setattr(durable_change_train_module, "reconcile_interrupted_durable_change_train", fake_recover)
    monkeypatch.setattr(durable_change_train_module, "capture_durable_database_evidence", fake_capture)
    monkeypatch.setattr(durable_change_train_module, "_historical_schema_evidence", lambda _train: None)
    monkeypatch.setattr(
        migration_runner,
        "capture_durable_schema_inventory",
        lambda _connection: SimpleNamespace(sha256="inventory"),
    )
    monkeypatch.setattr(
        durable_change_train_module,
        "_canonical_schema_inventory",
        lambda _tier, _version: SimpleNamespace(sha256="canonical"),
    )
    monkeypatch.setattr(
        durable_change_train_module,
        "_verify_released_train_live_tier",
        lambda _root, _connection, train, **_kwargs: events.append(("verify", train.target_version)),
    )

    durable_change_train_module._reconcile_durable_change_train_startup_locked(tmp_path)

    assert events == [("recover", later_slot), ("verify", first_slot), ("verify", later_slot)]


def test_startup_checks_chain_when_only_current_train_remains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_root = tmp_path / ".maintenance-state" / "durable-change-trains"
    manifest_root.mkdir(parents=True)
    manifest_path = manifest_root / f"source-{_NEXT_SOURCE_SLOT + 1:03d}.json"
    manifest_path.touch()
    current = cast(
        DurableChangeTrain,
        SimpleNamespace(
            state=DurableChangeTrainState.RELEASED, tier=ArchiveTier.SOURCE, target_version=_NEXT_SOURCE_SLOT + 1
        ),
    )

    @contextmanager
    def fake_open_tier(_path: Path) -> Iterator[sqlite3.Connection]:
        with sqlite3.connect(":memory:") as connection:
            yield connection

    monkeypatch.setattr(durable_change_train_module, "_open_existing_tier", fake_open_tier)
    monkeypatch.setattr(durable_change_train_module, "load_durable_change_train_manifest", lambda _path: current)
    monkeypatch.setattr(
        durable_change_train_module,
        "capture_durable_database_evidence",
        lambda _connection, _tier: SimpleNamespace(user_version=_NEXT_SOURCE_SLOT + 1),
    )
    monkeypatch.setattr(
        durable_change_train_module,
        "_released_train_manifests_by_target",
        lambda _root, _tier: {_NEXT_SOURCE_SLOT + 1: current},
    )

    with pytest.raises(DurableChangeTrainError, match="lacks released train evidence"):
        durable_change_train_module._reconcile_durable_change_train_startup_locked(tmp_path)


def test_startup_checks_chain_when_manifest_directory_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "source.db").touch()

    @contextmanager
    def fake_open_tier(_path: Path) -> Iterator[sqlite3.Connection]:
        with sqlite3.connect(":memory:") as connection:
            connection.execute(f"PRAGMA user_version = {_NEXT_SOURCE_SLOT}")
            yield connection

    monkeypatch.setattr(durable_change_train_module, "_open_existing_tier", fake_open_tier)

    with pytest.raises(DurableChangeTrainError, match="lacks released train evidence"):
        durable_change_train_module._reconcile_durable_change_train_startup_locked(tmp_path)


def test_fresh_archive_bootstrap_receipt_allows_repeat_startup(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    assert reconcile_durable_change_train_startup(tmp_path) == ()
    initialize_active_archive_root(tmp_path)


def test_audit_adoption_receipt_survives_startup_preflight(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The storage startup route validates the receipt created by the real adopter."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    (archive_root / "audit.db").unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    fsynced_paths: set[Path] = set()
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        try:
            fsynced_paths.add(Path(os.readlink(f"/proc/self/fd/{descriptor}")))
        except OSError:
            pass
        real_fsync(descriptor)

    monkeypatch.setattr("polylogue.operations.durable_change_train.os.fsync", record_fsync)

    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-adoption") as owner:
        version, receipt = adopt_missing_audit_tier(
            archive_root / "audit.db",
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    assert version == ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT]
    assert receipt == audit_adoption_receipt_path(archive_root)
    assert {
        archive_root,
        archive_root / ".maintenance-state",
        archive_root / ".maintenance-state" / "durable-change-trains",
    }.issubset(fsynced_paths)
    assert reconcile_durable_change_train_startup(archive_root) == ()
    receipt.write_text("tampered", encoding="utf-8")
    with pytest.raises(MigrationError, match="invalid audit adoption receipt"):
        reconcile_durable_change_train_startup(archive_root)


def test_audit_adoption_receipt_allows_a_mutated_audit_journal(workspace_env: dict[str, Path]) -> None:
    """Startup accepts an adopted audit tier after normal SQLite journal writes."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-mutable-journal") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    with closing(sqlite3.connect(audit_path)) as connection:
        connection.execute(
            "INSERT INTO archive_authority (archive_instance_id, created_at_ms, authority_format) VALUES (?, ?, ?)",
            ("adopted-audit-journal", 1, 1),
        )
        connection.commit()

    assert reconcile_durable_change_train_startup(archive_root) == ()


def test_audit_metadata_read_is_read_only_for_uri_metacharacter_paths(tmp_path: Path) -> None:
    """Archive path punctuation cannot consume SQLite's read-only URI parameter."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = tmp_path / "archive?#uri"
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"

    version, application_id, quick_check = _audit_live_metadata(audit_path)

    assert version == ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT]
    assert application_id == 0
    assert quick_check == ("ok",)
    assert not audit_path.with_name("audit.db-wal").exists()
    assert not audit_path.with_name("audit.db-shm").exists()


def test_adopted_audit_restore_rebinds_continuity_from_verified_backup(workspace_env: dict[str, Path]) -> None:
    """The real offline restore publishes a new immutable continuity generation."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(output_dir=archive_root.parent / "pre-adoption", profile="full_evidence", verify=True)
    assert pre_adoption.ok, pre_adoption.error
    assert pre_adoption.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-restore-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(output_dir=archive_root.parent / "post-adoption", profile="full_evidence", verify=True)
    assert verified.ok, verified.error
    assert verified.output_path is not None
    with closing(sqlite3.connect(archive_root / "index.db")) as connection:
        current_index_version = int(connection.execute("PRAGMA user_version").fetchone()[0] or 0)
        connection.execute(f"PRAGMA user_version = {current_index_version + 1}")
        connection.commit()
    old_identity = (audit_path.stat().st_dev, audit_path.stat().st_ino)
    audit_path.write_bytes(b"corrupted audit image")

    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-restore") as owner:
        receipt = restore_adopted_audit_tier(
            audit_path,
            backup_manifest=Path(verified.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    with closing(sqlite3.connect(audit_path)) as audit:
        assert audit.execute("PRAGMA user_version").fetchone() == (ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT],)
        assert audit.execute("SELECT generation FROM audit_continuity_head").fetchone() == (2,)
    assert (audit_path.stat().st_dev, audit_path.stat().st_ino) != old_identity
    assert receipt.name.endswith(".committed.json")
    assert receipt.with_name(receipt.name.replace(".committed.json", ".prepared.json")).is_file()
    with (
        closing(sqlite3.connect(archive_root / "source.db")) as source,
        closing(sqlite3.connect(archive_root / "audit.db")) as audit,
    ):
        assert (
            source.execute(
                "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control"
            ).fetchone()
            == audit.execute("SELECT generation, head_sha256 FROM audit_continuity_head").fetchone()
        )
    assert reconcile_durable_change_train_startup(archive_root) == ()


def test_adopted_audit_restore_resumes_an_interrupted_continuity_commit(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prepared record blocks startup but the same verified backup can complete it."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(output_dir=archive_root.parent / "resume-pre", profile="full_evidence", verify=True)
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-resume-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(output_dir=archive_root.parent / "resume-post", profile="full_evidence", verify=True)
    assert verified.ok and verified.output_path is not None, verified.error
    audit_path.write_bytes(b"corrupt")
    source_wal_reader: sqlite3.Connection | None = None
    from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator

    original_phase = AuditContinuityCoordinator._phase

    def interrupt_after_rebind_commit(self: AuditContinuityCoordinator, phase: str, mutation: object) -> None:
        nonlocal source_wal_reader
        if getattr(mutation, "mutation_id", "").startswith("audit-restore:"):
            if phase == "before_source_prepare":
                with sqlite3.connect(archive_root / "source.db") as source:
                    assert source.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
                source_wal_reader = sqlite3.connect(archive_root / "source.db")
                source_wal_reader.execute("BEGIN")
                source_wal_reader.execute("SELECT * FROM audit_continuity_control").fetchone()
            if phase == "after_source_prepare":
                raise RuntimeError("simulated continuity prepare interruption")
        original_phase(self, phase, mutation)  # type: ignore[arg-type]

    try:
        with monkeypatch.context() as interrupted:
            interrupted.setattr(AuditContinuityCoordinator, "_phase", interrupt_after_rebind_commit)
            with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-resume-interrupt") as owner:
                with pytest.raises(RuntimeError, match="continuity prepare interruption"):
                    restore_adopted_audit_tier(
                        audit_path,
                        backup_manifest=Path(verified.output_path) / "manifest.json",
                        directory_fd=owner.directory_fd,
                        stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                    )

        # The production source-prepare transition is durable only in a WAL;
        # retry must validate this operation-owned pending command before it
        # can republish/rebind the audit image.
        assert (archive_root / "source.db-wal").stat().st_size > 0
        with sqlite3.connect(archive_root / "source.db") as source, sqlite3.connect(audit_path) as audit:
            assert str(
                source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone()[0]
            ).startswith("audit-restore:")
            assert (
                source.execute(
                    "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control"
                ).fetchone()
                == audit.execute("SELECT generation, head_sha256 FROM audit_continuity_head").fetchone()
            )
        with sqlite3.connect(archive_root / "source.db") as source:
            source.execute("CREATE TABLE restore_retry_tamper (value TEXT NOT NULL) STRICT")
            source.commit()
        with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-resume-tamper") as owner:
            with pytest.raises(MigrationError, match="backup is stale for source.db"):
                restore_adopted_audit_tier(
                    audit_path,
                    backup_manifest=Path(verified.output_path) / "manifest.json",
                    directory_fd=owner.directory_fd,
                    stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                )
        with sqlite3.connect(archive_root / "source.db") as source:
            source.execute("DROP TABLE restore_retry_tamper")
            source.commit()
        with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-resume") as owner:
            receipt = restore_adopted_audit_tier(
                audit_path,
                backup_manifest=Path(verified.output_path) / "manifest.json",
                directory_fd=owner.directory_fd,
                stopped_daemon_check=lambda: "proof:test-daemon-stopped",
            )
    finally:
        if source_wal_reader is not None:
            source_wal_reader.close()

    assert receipt.name.endswith(".committed.json")
    assert reconcile_durable_change_train_startup(archive_root) == ()


def test_adopted_audit_restore_republishes_before_reading_a_promoted_unreadable_audit(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry restores its verified image before authenticating a promoted rebind head."""

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(
        output_dir=archive_root.parent / "promoted-missing-pre", profile="full_evidence", verify=True
    )
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id="test:promoted-missing-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(
        output_dir=archive_root.parent / "promoted-missing-post", profile="full_evidence", verify=True
    )
    assert verified.ok and verified.output_path is not None, verified.error
    audit_path.write_bytes(b"corrupt before promoted crash")
    original_phase = AuditContinuityCoordinator._phase

    def crash_after_source_promotion(self: AuditContinuityCoordinator, phase: str, mutation: object) -> None:
        if getattr(mutation, "mutation_id", "").startswith("audit-restore:") and phase == "after_source_promotion":
            raise RuntimeError("crash after restore source promotion")
        original_phase(self, phase, mutation)  # type: ignore[arg-type]

    with monkeypatch.context() as interrupted:
        interrupted.setattr(AuditContinuityCoordinator, "_phase", crash_after_source_promotion)
        with acquire_durable_archive_ownership(archive_root, owner_id="test:promoted-missing-crash") as owner:
            with pytest.raises(RuntimeError, match="crash after restore source promotion"):
                restore_adopted_audit_tier(
                    audit_path,
                    backup_manifest=Path(verified.output_path) / "manifest.json",
                    directory_fd=owner.directory_fd,
                    stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                )

    with sqlite3.connect(archive_root / "source.db") as source:
        promoted_source_tuple = source.execute(
            "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control"
        ).fetchone()

    # The source promotion survived while the live authority image became
    # unreadable. The retry must publish the verified backup before reading
    # its continuity head.
    audit_path.write_bytes(b"unreadable after promoted restore crash")
    with acquire_durable_archive_ownership(archive_root, owner_id="test:promoted-missing-retry") as owner:
        receipt = restore_adopted_audit_tier(
            audit_path,
            backup_manifest=Path(verified.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    assert receipt.name.endswith(".committed.json")
    with sqlite3.connect(archive_root / "source.db") as source, sqlite3.connect(audit_path) as audit:
        assert (
            source.execute(
                "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control"
            ).fetchone()
            == promoted_source_tuple
        )
        assert (
            source.execute(
                "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control"
            ).fetchone()
            == audit.execute("SELECT generation, head_sha256 FROM audit_continuity_head").fetchone()
        )


def test_adopted_audit_restore_rejects_an_unrelated_higher_promoted_source_head(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cleared pending row cannot authorize an arbitrary higher source generation."""

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(
        output_dir=archive_root.parent / "higher-head-pre", profile="full_evidence", verify=True
    )
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id="test:higher-head-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(output_dir=archive_root.parent / "higher-head-post", profile="full_evidence", verify=True)
    assert verified.ok and verified.output_path is not None, verified.error
    audit_path.write_bytes(b"corrupt before forged higher promoted head")
    original_phase = AuditContinuityCoordinator._phase

    def crash_after_source_promotion(self: AuditContinuityCoordinator, phase: str, mutation: object) -> None:
        if getattr(mutation, "mutation_id", "").startswith("audit-restore:") and phase == "after_source_promotion":
            raise RuntimeError("crash after restore source promotion")
        original_phase(self, phase, mutation)  # type: ignore[arg-type]

    with monkeypatch.context() as interrupted:
        interrupted.setattr(AuditContinuityCoordinator, "_phase", crash_after_source_promotion)
        with acquire_durable_archive_ownership(archive_root, owner_id="test:higher-head-crash") as owner:
            with pytest.raises(RuntimeError, match="crash after restore source promotion"):
                restore_adopted_audit_tier(
                    audit_path,
                    backup_manifest=Path(verified.output_path) / "manifest.json",
                    directory_fd=owner.directory_fd,
                    stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                )

    with sqlite3.connect(archive_root / "source.db") as source:
        source.execute(
            "UPDATE audit_continuity_control SET committed_generation = committed_generation + 7, "
            "committed_head_sha256 = ? WHERE singleton = 1",
            ("f" * 64,),
        )
        source.commit()
    audit_path.write_bytes(b"unreadable after forged higher promoted head")

    with acquire_durable_archive_ownership(archive_root, owner_id="test:higher-head-retry") as owner:
        with pytest.raises(MigrationError, match="rebind is not operation-owned"):
            restore_adopted_audit_tier(
                audit_path,
                backup_manifest=Path(verified.output_path) / "manifest.json",
                directory_fd=owner.directory_fd,
                stopped_daemon_check=lambda: "proof:test-daemon-stopped",
            )


@pytest.mark.parametrize(
    ("entry_name", "entry_kind"),
    (
        ("audit.db", "directory"),
        ("audit.db", "dangling_symlink"),
        ("source.db", "symlink"),
    ),
)
def test_precontinuity_binding_rejects_invalid_present_archive_entries(
    tmp_path: Path, entry_name: str, entry_kind: str
) -> None:
    """Only truly absent durable entries can leave pre-continuity binding in standby."""

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    source_path = tmp_path / "source.db"
    entry_path = tmp_path / entry_name
    with closing(sqlite3.connect(source_path)) as source:
        if entry_kind == "directory":
            entry_path.unlink()
            entry_path.mkdir()
        elif entry_kind == "dangling_symlink":
            entry_path.unlink()
            entry_path.symlink_to(tmp_path / "missing-audit.db")
        else:
            external = tmp_path.parent / "external-source.db"
            external.write_bytes(entry_path.read_bytes())
            entry_path.unlink()
            entry_path.symlink_to(external)

        with pytest.raises(MigrationError, match="invalid pre-continuity"):
            migration_runner._bind_populated_precontinuity_audit(source, backup_manifest=None)


def test_adopted_audit_restore_replaces_stale_operation_staging_after_crash(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One retry completes a prepared restore after a crash leaves its private image."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(output_dir=archive_root.parent / "staging-pre", profile="full_evidence", verify=True)
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-staging-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(output_dir=archive_root.parent / "staging-post", profile="full_evidence", verify=True)
    assert verified.ok and verified.output_path is not None, verified.error
    audit_path.write_bytes(b"corrupt")
    real_unlink = os.unlink

    def interrupt_publication(*args: object, **kwargs: object) -> None:
        raise OSError("simulated restore publication crash")

    def leave_staging(name: os.PathLike[str] | str, *, dir_fd: int | None = None) -> None:
        if str(name).startswith(".audit.db.restore-"):
            return
        real_unlink(name, dir_fd=dir_fd)

    with monkeypatch.context() as interrupted:
        interrupted.setattr("polylogue.operations.durable_change_train.os.replace", interrupt_publication)
        interrupted.setattr("polylogue.operations.durable_change_train.os.unlink", leave_staging)
        with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-staging-interrupt") as owner:
            with pytest.raises(OSError, match="simulated restore publication crash"):
                restore_adopted_audit_tier(
                    audit_path,
                    backup_manifest=Path(verified.output_path) / "manifest.json",
                    directory_fd=owner.directory_fd,
                    stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                )

    stale = tuple(archive_root.glob(".audit.db.restore-*.tmp"))
    assert len(stale) == 1
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-staging-resume") as owner:
        receipt = restore_adopted_audit_tier(
            audit_path,
            backup_manifest=Path(verified.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    assert receipt.name.endswith(".committed.json")
    assert not tuple(archive_root.glob(".audit.db.restore-*.tmp"))
    assert reconcile_durable_change_train_startup(archive_root) == ()


def test_adopted_audit_restore_record_survives_publication_temp_hardlink(
    workspace_env: dict[str, Path],
) -> None:
    """A crash after immutable publication may leave the valid record with two names."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(output_dir=archive_root.parent / "hardlink-pre", profile="full_evidence", verify=True)
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-hardlink-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(output_dir=archive_root.parent / "hardlink-post", profile="full_evidence", verify=True)
    assert verified.ok and verified.output_path is not None, verified.error
    audit_path.write_bytes(b"corrupt")
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-hardlink-restore") as owner:
        committed = restore_adopted_audit_tier(
            audit_path,
            backup_manifest=Path(verified.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    leftover = committed.with_name(f".{committed.name}.publication.tmp")
    os.link(committed, leftover)

    assert committed.stat().st_nlink == 2
    assert reconcile_durable_change_train_startup(archive_root) == ()


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    [
        ("artifact", "migration backup tier artifact"),
        ("receipt", "adopted-audit restore"),
        ("stale-source", "adopted-audit restore backup is stale for source.db"),
    ],
)
def test_adopted_audit_restore_rejects_untrusted_or_stale_backup(
    workspace_env: dict[str, Path], tamper: str, expected_error: str
) -> None:
    """Mutation: bypassing receipt, artifact, or current-authority checks reaches the real restore call."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(
        output_dir=archive_root.parent / f"pre-{tamper}", profile="full_evidence", verify=True
    )
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id=f"test:audit-restore-adopt-{tamper}") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(output_dir=archive_root.parent / f"post-{tamper}", profile="full_evidence", verify=True)
    assert verified.ok and verified.output_path is not None, verified.error
    backup_root = Path(verified.output_path)
    if tamper == "artifact":
        (backup_root / "audit.db").write_bytes(b"altered backup audit")
    elif tamper == "receipt":
        (backup_root / "verification-receipt.json").write_text("{}", encoding="utf-8")
    else:
        with closing(sqlite3.connect(archive_root / "source.db")) as connection:
            connection.execute("PRAGMA user_version = 999")
            connection.commit()
    audit_path.write_bytes(b"corrupted audit image")

    with acquire_durable_archive_ownership(archive_root, owner_id=f"test:audit-restore-reject-{tamper}") as owner:
        with pytest.raises(MigrationError, match=expected_error):
            restore_adopted_audit_tier(
                audit_path,
                backup_manifest=backup_root / "manifest.json",
                directory_fd=owner.directory_fd,
                stopped_daemon_check=lambda: "proof:test-daemon-stopped",
            )

    assert audit_path.read_bytes() == b"corrupted audit image"


def test_adopted_audit_restore_rejects_version_skew(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restore admission rejects a staged audit image at the wrong durable version.

    The admission at ``restore_adopted_audit_tier`` requires the staged image
    to stand at the audit tier's *current* durable version, not merely at the
    version its own receipt claims. An image one slot away is refused before
    it is copied into the owned root, and the live adopted tier is untouched.

    The pre-reset version of this test skewed the staged image by dropping
    ``audit_continuity_head`` and stamping ``user_version = 1``; with the
    audit floor now at 1 that stamp is the *current* version, so the fixture
    no longer produced any skew and the admission under test was reached with
    nothing to reject. The version leg is expressed against
    ``ARCHIVE_VERSION_BY_TIER`` here rather than a literal, so it cannot go
    stale the same way again.

    Anti-vacuity: drop ``backup_version != ARCHIVE_VERSION_BY_TIER[AUDIT]``
    from the admission in ``restore_adopted_audit_tier`` and the skewed image
    is admitted -- the call no longer raises. The receipt below carries the
    staged image's real digest and size, so the copy guard cannot stand in
    for the admission and produce a green test for the wrong reason.
    """

    from polylogue.operations import durable_change_train as operations_durable_change_train
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(
        output_dir=archive_root.parent / "version-skew-pre", profile="full_evidence", verify=True
    )
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id="test:version-skew-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(
        output_dir=archive_root.parent / "version-skew-post", profile="full_evidence", verify=True
    )
    assert verified.ok and verified.output_path is not None, verified.error
    backup_root = Path(verified.output_path)
    skewed_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT] + 1
    with sqlite3.connect(backup_root / "audit.db") as staged:
        staged.execute(f"PRAGMA user_version = {skewed_version}")
    # The staged tier is in WAL mode and admission reads it with
    # ``immutable=1``. Without this checkpoint the stamp stays in the WAL, the
    # reader still sees the current version, and the refusal below fires on
    # the receipt-vs-file leg instead of the one under test -- which is
    # exactly how this test survived its own mutation.
    checkpoint_durable_tier(backup_root / "audit.db")
    staged_bytes = (backup_root / "audit.db").read_bytes()
    with closing(sqlite3.connect(f"file:{backup_root / 'audit.db'}?immutable=1", uri=True)) as observed:
        assert int(observed.execute("PRAGMA user_version").fetchone()[0]) == skewed_version
    receipt = archive_root.parent / "version-skew-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "tier_artifacts": [
                    {
                        "tier": "audit",
                        "sha256": hashlib.sha256(staged_bytes).hexdigest(),
                        "size_bytes": len(staged_bytes),
                        "user_version": skewed_version,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        operations_durable_change_train,
        "validate_full_evidence_backup_for_adopted_audit_restore",
        lambda *_args, **_kwargs: (backup_root / "manifest.json", receipt),
    )

    with acquire_durable_archive_ownership(archive_root, owner_id="test:version-skew-restore") as owner:
        with pytest.raises(MigrationError, match="does not belong"):
            restore_adopted_audit_tier(
                audit_path,
                backup_manifest=backup_root / "manifest.json",
                directory_fd=owner.directory_fd,
                stopped_daemon_check=lambda: "proof:test-daemon-stopped",
            )

    with sqlite3.connect(audit_path) as audit:
        assert audit.execute("PRAGMA user_version").fetchone() == (ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT],)
        assert audit.execute("SELECT 1 FROM audit_continuity_head").fetchone() == (1,)


def test_adopted_audit_restore_rejects_wrong_archive_application_id(
    workspace_env: dict[str, Path],
) -> None:
    """A valid backup from different audit authority cannot replace the adopted journal."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(output_dir=archive_root.parent / "app-id-pre", profile="full_evidence", verify=True)
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-app-id-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    with closing(sqlite3.connect(audit_path)) as connection:
        adopted_application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
        connection.execute(f"PRAGMA application_id = {adopted_application_id + 1}")
        connection.commit()
    wrong_authority = backup_archive(
        output_dir=archive_root.parent / "app-id-wrong", profile="full_evidence", verify=True
    )
    assert wrong_authority.ok and wrong_authority.output_path is not None, wrong_authority.error
    with closing(sqlite3.connect(audit_path)) as connection:
        connection.execute(f"PRAGMA application_id = {adopted_application_id}")
        connection.commit()
    original = audit_path.read_bytes()

    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-app-id-restore") as owner:
        with pytest.raises(MigrationError, match="does not belong to this audit adoption"):
            restore_adopted_audit_tier(
                audit_path,
                backup_manifest=Path(wrong_authority.output_path) / "manifest.json",
                directory_fd=owner.directory_fd,
                stopped_daemon_check=lambda: "proof:test-daemon-stopped",
            )

    assert audit_path.read_bytes() == original


def test_adopted_audit_restore_rejects_backup_swap_after_validation(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact manifest and verification receipt stay fixed through publication."""
    from polylogue.operations import durable_change_train as operations_durable_change_train
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(output_dir=archive_root.parent / "swap-pre", profile="full_evidence", verify=True)
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-swap-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(output_dir=archive_root.parent / "swap-post", profile="full_evidence", verify=True)
    assert verified.ok and verified.output_path is not None, verified.error
    backup_root = Path(verified.output_path)
    audit_path.write_bytes(b"corrupt-before-swap-test")
    from polylogue.storage.sqlite.migration_runner import validate_full_evidence_backup_for_adopted_audit_restore

    real_validate = validate_full_evidence_backup_for_adopted_audit_restore
    calls = 0

    def swap_after_validation(path: Path, *, archive_root: Path, **kwargs: object) -> tuple[Path, Path]:
        nonlocal calls
        calls += 1
        manifest, receipt = real_validate(path, archive_root=archive_root, **kwargs)  # type: ignore[arg-type]
        if calls == 2:
            receipt.write_bytes(receipt.read_bytes() + b"\n")
        return manifest, receipt

    monkeypatch.setattr(
        operations_durable_change_train,
        "validate_full_evidence_backup_for_adopted_audit_restore",
        swap_after_validation,
    )
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-swap-restore") as owner:
        with pytest.raises(MigrationError, match="backup changed during the operation"):
            restore_adopted_audit_tier(
                audit_path,
                backup_manifest=backup_root / "manifest.json",
                directory_fd=owner.directory_fd,
                stopped_daemon_check=lambda: "proof:test-daemon-stopped",
            )

    assert audit_path.read_bytes() == b"corrupt-before-swap-test"


def test_audit_adoption_binds_only_the_source_user_authority(workspace_env: dict[str, Path]) -> None:
    """Routine replacement of rebuildable or disposable tiers leaves adoption valid."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-stable-authority") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    (archive_root / "index.db").unlink()
    (archive_root / "ops.db").unlink()
    initialize_active_archive_root(archive_root)

    assert (archive_root / "index.db").is_file()
    assert (archive_root / "ops.db").is_file()


def test_audit_adoption_continuity_survives_index_generation_promotion(
    workspace_env: dict[str, Path],
) -> None:
    """Promotion of a new physical index root does not invalidate audit adoption."""
    from polylogue.storage.archive_identity import ArchiveIdentity
    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-promotion-authority") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    durable_authority_before = ArchiveIdentity.resolve(archive_root).durable_id
    adoption_receipt = json.loads(audit_adoption_receipt_path(archive_root).read_text(encoding="utf-8"))

    generation_store = IndexGenerationStore.for_archive_root(archive_root)
    candidate = generation_store.create(source_snapshot="audit-adoption-promotion-proof")
    promoted = generation_store.promote(candidate)

    assert promoted.state == "active"
    assert ArchiveIdentity.resolve(archive_root).durable_id == durable_authority_before
    assert validate_audit_adoption_receipt(archive_root) == audit_adoption_receipt_path(archive_root)
    assert json.loads(audit_adoption_receipt_path(archive_root).read_text(encoding="utf-8")) == adoption_receipt
    initialize_active_archive_root(archive_root)


def test_audit_adoption_receipt_keeps_initial_schema_evidence_after_upgrade(
    workspace_env: dict[str, Path],
) -> None:
    """Receipt validation permits later audit schema versions for normal train handling."""
    from polylogue.operations.durable_change_train import validate_audit_adoption_receipt
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-schema-evidence") as owner:
        _version, receipt = adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    initial_schema_digest = json.loads(receipt.read_text(encoding="utf-8"))["audit_schema_inventory_sha256"]

    # "Later" is derived from the live audit DDL, never a literal: the previous
    # literal silently became a *downgrade* the moment audit gained migration
    # 003, and the refusal that produced was read as this test's own failure.
    future_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT] + 1
    with closing(sqlite3.connect(audit_path)) as connection:
        connection.execute("CREATE TABLE future_audit_schema (value TEXT)")
        connection.execute(f"PRAGMA user_version = {future_version}")
        connection.commit()

    assert validate_audit_adoption_receipt(archive_root) == receipt
    assert json.loads(receipt.read_text(encoding="utf-8"))["audit_schema_inventory_sha256"] == initial_schema_digest


def test_audit_adoption_rejects_a_stale_audit_file_clone(workspace_env: dict[str, Path]) -> None:
    """The continuity record distinguishes in-place writes from a stale file replacement."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-stale-clone") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    stale_clone = archive_root / "stale-audit.db"
    shutil.copy2(audit_path, stale_clone)
    with closing(sqlite3.connect(audit_path)) as connection:
        connection.execute(
            "INSERT INTO archive_authority (archive_instance_id, created_at_ms, authority_format) VALUES (?, ?, ?)",
            ("live-audit-after-clone", 2, 1),
        )
        connection.commit()
    os.replace(stale_clone, audit_path)

    with pytest.raises(MigrationError, match="continuity"):
        reconcile_durable_change_train_startup(archive_root)


def test_adopted_audit_receipt_is_checked_once(workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """Startup reconciliation validates the adoption receipt exactly once.

    The immutable adoption receipt is the authority that binds an adopted
    audit tier to the backup it came from, and re-reading it per tier or per
    manifest would make its cost scale with the archive. One check per
    reconciliation pass is the contract.

    The check is driven through ``reconcile_durable_change_trains_on_startup``
    rather than ``initialize_active_archive_root``: #5275 made archive
    bootstrap skip startup reconciliation for a format-marked archive, so
    bootstrap now performs zero receipt checks and the reconciler -- the route
    the daemon runs -- owns this one.

    Anti-vacuity: move the ``validate_audit_adoption_receipt`` call inside the
    per-manifest loop of ``_reconcile_durable_change_train_startup_locked``
    and the count goes above one; delete it and the count goes to zero.
    """
    from polylogue.operations import durable_change_train as operations_durable_change_train
    from polylogue.operations.durable_change_train import reconcile_durable_change_trains_on_startup
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-single-quick-check") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    calls = 0
    real_validate = operations_durable_change_train.validate_audit_adoption_receipt

    def count_validate(root: Path, *, require_initial_image: bool = False) -> Path | None:
        nonlocal calls
        calls += 1
        return real_validate(root, require_initial_image=require_initial_image)

    monkeypatch.setattr(operations_durable_change_train, "validate_audit_adoption_receipt", count_validate)
    reconcile_durable_change_trains_on_startup(archive_root)

    assert calls == 1


def test_audit_adoption_receipt_recovers_interrupted_publication_during_bootstrap(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The normal bootstrap route completes the receipt-backed publication after a crash."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    marker_root = archive_root / ".maintenance-state" / "durable-change-trains"
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    monkeypatch.setitem(
        durable_change_train_module.DURABLE_MIGRATION_ADOPTION_FLOORS,
        ArchiveTier.SOURCE,
        ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] - 1,
    )
    monkeypatch.setitem(
        durable_change_train_module.DURABLE_MIGRATION_ADOPTION_FLOORS,
        ArchiveTier.USER,
        ARCHIVE_VERSION_BY_TIER[ArchiveTier.USER] - 1,
    )
    real_link = os.link

    def fail_audit_link(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(destination).name == "audit.db":
            raise OSError("simulated interruption")
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    with monkeypatch.context() as failed_publication:
        failed_publication.setattr("polylogue.operations.durable_change_train.os.link", fail_audit_link)
        with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-interrupted-publication") as owner:
            with pytest.raises(MigrationError):
                adopt_missing_audit_tier(
                    audit_path,
                    backup_manifest=Path(backup.output_path) / "manifest.json",
                    directory_fd=owner.directory_fd,
                    stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                )

    initialize_active_archive_root(archive_root)

    assert audit_path.is_file()
    assert (marker_root / ".bootstrap").is_file()
    assert reconcile_durable_change_train_startup(archive_root) == ()


def test_audit_adoption_retry_reports_recovered_audit_schema_version(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt-backed retry reports the live audit schema, not a sentinel."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok and backup.output_path is not None, backup.error
    manifest = Path(backup.output_path) / "manifest.json"
    from polylogue.operations import durable_change_train as operations_durable_change_train

    real_validate = operations_durable_change_train.validate_audit_adoption_receipt

    def interrupt_after_publication(root: Path, *, require_initial_image: bool = False) -> Path | None:
        if require_initial_image:
            raise RuntimeError("simulated post-publication interruption")
        return real_validate(root, require_initial_image=require_initial_image)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(
            operations_durable_change_train, "validate_audit_adoption_receipt", interrupt_after_publication
        )
        with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-adoption") as owner:
            with pytest.raises(RuntimeError, match="post-publication interruption"):
                adopt_missing_audit_tier(
                    audit_path,
                    backup_manifest=manifest,
                    directory_fd=owner.directory_fd,
                    stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                )
    assert audit_path.is_file()

    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-adoption-retry") as owner:
        recovered_version, _receipt = adopt_missing_audit_tier(
            audit_path,
            backup_manifest=manifest,
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    assert recovered_version == ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT]


def test_audit_adoption_rejects_a_stale_replacement(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup does not bless a replaced audit image in the post-link crash window.

    Adoption links the audit tier and then publishes ``audit-continuity.json``.
    A crash between the two leaves a receipt whose audit image is already
    live; swapping that image for an older copy in the gap must be refused,
    the stale bytes left exactly as found, and no continuity record written.

    Driven through ``reconcile_durable_change_trains_on_startup``: the refusal
    is raised by ``validate_audit_adoption_receipt``, which #5275 removed from
    the bootstrap path for a format-marked archive.

    Anti-vacuity: drop the live-image comparison from
    ``validate_audit_adoption_receipt`` and the stale clone is accepted --
    the call returns and ``audit-continuity.json`` appears.
    """
    from polylogue.operations.durable_change_train import reconcile_durable_change_trains_on_startup
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    marker_root = archive_root / ".maintenance-state" / "durable-change-trains"
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    real_link = os.link

    def interrupt_continuity_link(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(destination).name == "audit-continuity.json":
            raise OSError("simulated crash after audit link")
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    with monkeypatch.context() as interrupted_publication:
        interrupted_publication.setattr("polylogue.operations.durable_change_train.os.link", interrupt_continuity_link)
        with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-continuity-crash") as owner:
            with pytest.raises(MigrationError, match="cannot publish immutable audit adoption receipt"):
                adopt_missing_audit_tier(
                    audit_path,
                    backup_manifest=Path(backup.output_path) / "manifest.json",
                    directory_fd=owner.directory_fd,
                    stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                )

    with sqlite3.connect(archive_root / "source.db") as source, sqlite3.connect(audit_path) as audit:
        source_head = source.execute(
            "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control"
        ).fetchone()
        audit_head = audit.execute("SELECT generation, head_sha256 FROM audit_continuity_head").fetchone()
    assert source_head == audit_head
    assert source_head[0] == 1

    stale_clone = archive_root / "stale-audit.db"
    with closing(sqlite3.connect(audit_path)) as connection:
        connection.execute(
            "INSERT INTO archive_authority (archive_instance_id, created_at_ms, authority_format) VALUES (?, ?, ?)",
            ("audit-before-stale-clone", 1, 1),
        )
        connection.commit()
    shutil.copy2(audit_path, stale_clone)
    with closing(sqlite3.connect(audit_path)) as connection:
        connection.execute(
            "INSERT INTO archive_authority (archive_instance_id, created_at_ms, authority_format) VALUES (?, ?, ?)",
            ("audit-after-stale-clone", 2, 1),
        )
        connection.commit()
    os.replace(stale_clone, audit_path)
    stale_image = audit_path.read_bytes()

    with pytest.raises(MigrationError, match="audit tier changed before recording adoption continuity"):
        reconcile_durable_change_trains_on_startup(archive_root)

    assert audit_path.read_bytes() == stale_image
    assert not (marker_root / "audit-continuity.json").exists()


def test_audit_adoption_resumes_a_seeded_head(workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """A continuity-receipt crash resumes the already-seeded adoption head.

    The machine head is seeded into both source and audit before the
    continuity receipt is published. A crash in that window must not re-seed
    or refuse; the next reconciliation pass finishes the publication it left.

    Driven through ``reconcile_durable_change_trains_on_startup``, which is
    where ``validate_audit_adoption_receipt`` now runs for a format-marked
    archive (#5275).

    Anti-vacuity: make ``validate_audit_adoption_receipt`` return without
    publishing when a seeded head already exists and
    ``audit-continuity.json`` never appears.
    """
    from polylogue.operations.durable_change_train import reconcile_durable_change_trains_on_startup
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(
        output_dir=archive_root.parent / "seed-before-publish", profile="full_evidence", verify=True
    )
    assert backup.ok and backup.output_path is not None, backup.error
    real_link = os.link

    def interrupt_continuity_link(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        **kwargs: object,
    ) -> None:
        if Path(destination).name == "audit-continuity.json":
            raise OSError("simulated crash after machine-head seed")
        real_link(source, destination, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as interrupted:
        interrupted.setattr("polylogue.operations.durable_change_train.os.link", interrupt_continuity_link)
        with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-seed-before-publish") as owner:
            with pytest.raises(MigrationError, match="cannot publish immutable audit adoption receipt"):
                adopt_missing_audit_tier(
                    audit_path,
                    backup_manifest=Path(backup.output_path) / "manifest.json",
                    directory_fd=owner.directory_fd,
                    stopped_daemon_check=lambda: "proof:test-daemon-stopped",
                )

    with sqlite3.connect(archive_root / "source.db") as source, sqlite3.connect(audit_path) as audit:
        assert source.execute("SELECT committed_generation FROM audit_continuity_control").fetchone() == (1,)
        assert audit.execute("SELECT generation FROM audit_continuity_head").fetchone() == (1,)
    reconcile_durable_change_trains_on_startup(archive_root)
    assert (archive_root / ".maintenance-state" / "durable-change-trains" / "audit-continuity.json").is_file()


def test_adopted_audit_restore_removes_owned_sidecars_before_publication(workspace_env: dict[str, Path]) -> None:
    """The real restore cannot replay stale audit WAL, SHM, or rollback bytes."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    pre_adoption = backup_archive(output_dir=archive_root.parent / "sidecar-pre", profile="full_evidence", verify=True)
    assert pre_adoption.ok and pre_adoption.output_path is not None, pre_adoption.error
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-sidecar-adopt") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(pre_adoption.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    verified = backup_archive(output_dir=archive_root.parent / "sidecar-post", profile="full_evidence", verify=True)
    assert verified.ok and verified.output_path is not None, verified.error
    audit_path.write_bytes(b"corrupt-audit-main")
    for suffix in ("-wal", "-shm", "-journal"):
        audit_path.with_name(f"audit.db{suffix}").write_bytes(b"stale-owned-sidecar")

    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-sidecar-restore") as owner:
        restore_adopted_audit_tier(
            audit_path,
            backup_manifest=Path(verified.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    assert not any(audit_path.with_name(f"audit.db{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))
    assert _audit_live_metadata(audit_path)[2] == ("ok",)


def test_audit_adoption_recovery_preserves_missing_tier_after_continuity(
    workspace_env: dict[str, Path],
) -> None:
    """Recovery requires restore, without recreating audit.db after completed adoption."""
    from polylogue.operations.durable_change_train import recover_pending_audit_adoption
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-missing-after-continuity") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    audit_path.unlink()
    before = {path.relative_to(archive_root): path.read_bytes() for path in archive_root.rglob("*") if path.is_file()}

    with pytest.raises(MigrationError, match="missing after continuity"):
        recover_pending_audit_adoption(archive_root)

    after = {path.relative_to(archive_root): path.read_bytes() for path in archive_root.rglob("*") if path.is_file()}
    assert after == before
    assert not audit_path.exists()


def test_audit_adoption_receipt_is_excluded_from_pre_marker_train_state(workspace_env: dict[str, Path]) -> None:
    """The adoption receipt does not prevent legacy current-schema bootstrap adoption."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    marker = archive_root / ".maintenance-state" / "durable-change-trains" / ".bootstrap"
    marker.unlink()
    audit_path = archive_root / "audit.db"
    audit_path.unlink()
    backup = backup_archive(output_dir=archive_root.parent / "backup", profile="full_evidence", verify=True)
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id="test:audit-pre-marker") as owner:
        adopt_missing_audit_tier(
            audit_path,
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )

    initialize_active_archive_root(archive_root)

    assert marker.is_file()


def test_adoption_receipt_short_write_is_removed_for_a_safe_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed receipt write leaves no immutable-looking truncated publication behind."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    receipt_path = audit_adoption_receipt_path(archive_root)
    write_calls = 0

    def short_then_fail(descriptor: int, data: bytes) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return min(1, len(data))
        raise OSError("simulated receipt write failure")

    monkeypatch.setattr("polylogue.operations.durable_change_train.os.write", short_then_fail)

    with pytest.raises(MigrationError, match="cannot publish immutable audit adoption receipt"):
        _write_immutable_audit_adoption_receipt(receipt_path, {"format": "test"}, archive_root=archive_root)

    assert not receipt_path.exists()


def test_adoption_receipt_refuses_a_symlinked_maintenance_parent(tmp_path: Path) -> None:
    """Receipt publication and loading stay beneath the owned archive descriptor."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (archive_root / ".maintenance-state").symlink_to(outside, target_is_directory=True)
    receipt_path = audit_adoption_receipt_path(archive_root)

    with pytest.raises(MigrationError, match="must not traverse outside archive-owned directories"):
        _write_immutable_audit_adoption_receipt(receipt_path, {"format": "test"}, archive_root=archive_root)

    assert not (outside / "durable-change-trains" / "audit-adoption.json").exists()


def test_runtime_bootstrap_refuses_an_established_archive_missing_audit(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordinary writable startup cannot create audit.db without adoption evidence."""
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    archive_root = workspace_env["archive_root"]
    bootstrap.initialize_active_archive_root(archive_root)
    (archive_root / "audit.db").unlink()
    reconciled = False

    def observe_reconciliation(_root: Path) -> tuple[Path, ...]:
        nonlocal reconciled
        reconciled = True
        return ()

    monkeypatch.setattr(bootstrap, "reconcile_durable_change_trains_on_startup", observe_reconciliation)

    with pytest.raises(RuntimeError, match="adopt-established-audit"):
        bootstrap.initialize_active_archive_root(archive_root)

    assert not reconciled
    assert not (archive_root / "audit.db").exists()


def test_pre_slot_source_missing_audit_names_adoption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lineage member one slot behind, without audit.db, names its recovery route.

    A source tier standing below the runtime target is ordinary: the archive
    was born before that tier's numbered slot shipped. Losing ``audit.db`` on
    top of that must still produce the adoption instruction an operator can
    act on -- not a schema complaint, and never a silently recreated audit
    tier over durable evidence nobody has.

    The pre-reset version stamped ``user_version = 31`` and reconstructed a
    v31 schema. Neither exists now, and 31 is above every version this lineage
    can hold, so ``assert_archive_format_lineage`` skipped its fingerprint
    comparison entirely and the test passed without describing a real archive.
    The fixture is bootstrapped at the floor through the production route and
    then reduced to the pre-slot-002 shape, with the marker restated, so the
    archive it hands bootstrap is one this runtime could actually have
    written.

    Anti-vacuity: remove the ``established_pair_without_audit`` branch from
    ``_initialize_active_archive_root`` and the second bootstrap recreates
    ``audit.db`` instead of refusing, turning both assertions red.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    _pin_source_runtime_version(monkeypatch, _SOURCE_ADOPTION_FLOOR)
    initialize_active_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source:
        reset_source_fixture_to_version(source, _SOURCE_ADOPTION_FLOOR)
        source.execute("DROP TABLE IF EXISTS audit_continuity_control")
        source.execute(f"PRAGMA user_version = {_SOURCE_ADOPTION_FLOOR}")
        source.commit()
    refresh_archive_format_marker(tmp_path)
    (tmp_path / "audit.db").unlink()

    with pytest.raises(RuntimeError, match="adopt-established-audit"):
        initialize_active_archive_root(tmp_path)

    assert not (tmp_path / "audit.db").exists()


def test_fresh_bootstrap_intent_recovers_after_late_tier_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    real_initialize_archive_database = bootstrap.initialize_archive_database
    failed = False

    def fail_embeddings_once(
        path: Path,
        tier: ArchiveTier,
        *,
        allow_create: bool = True,
        expected_version: int | None = None,
    ) -> None:
        nonlocal failed
        if tier is ArchiveTier.EMBEDDINGS and not failed:
            failed = True
            raise RuntimeError("simulated late fresh-bootstrap failure")
        real_initialize_archive_database(path, tier, allow_create=allow_create, expected_version=expected_version)

    monkeypatch.setattr(bootstrap, "initialize_archive_database", fail_embeddings_once)
    with pytest.raises(RuntimeError, match="simulated late fresh-bootstrap failure"):
        bootstrap.initialize_active_archive_root(tmp_path)

    marker_root = tmp_path / ".maintenance-state" / "durable-change-trains"
    assert (marker_root / ".bootstrap.pending").is_file()
    assert not (marker_root / ".bootstrap").exists()
    assert (tmp_path / "source.db").is_file()

    monkeypatch.setattr(bootstrap, "initialize_archive_database", real_initialize_archive_database)
    bootstrap.initialize_active_archive_root(tmp_path)

    assert (marker_root / ".bootstrap").is_file()
    assert not (marker_root / ".bootstrap.pending").exists()
    assert reconcile_durable_change_train_startup(tmp_path) == ()


def test_fresh_bootstrap_intent_rejects_tampering_before_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    real_initialize_archive_database = bootstrap.initialize_archive_database

    def fail_embeddings(
        path: Path,
        tier: ArchiveTier,
        *,
        allow_create: bool = True,
        expected_version: int | None = None,
    ) -> None:
        if tier is ArchiveTier.EMBEDDINGS:
            raise RuntimeError("simulated late fresh-bootstrap failure")
        real_initialize_archive_database(path, tier, allow_create=allow_create, expected_version=expected_version)

    monkeypatch.setattr(bootstrap, "initialize_archive_database", fail_embeddings)
    with pytest.raises(RuntimeError, match="simulated late fresh-bootstrap failure"):
        bootstrap.initialize_active_archive_root(tmp_path)

    pending = tmp_path / ".maintenance-state" / "durable-change-trains" / ".bootstrap.pending"
    payload = json.loads(pending.read_text(encoding="utf-8"))
    payload["durable_identity_digest"] = "0" * 64
    pending.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DurableChangeTrainError, match="intent durable identity mismatch"):
        bootstrap.initialize_active_archive_root(tmp_path)


def test_pre_marker_current_archive_is_adopted_once(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    marker = tmp_path / ".maintenance-state" / "durable-change-trains" / ".bootstrap"
    marker.unlink()

    initialize_active_archive_root(tmp_path)
    assert marker.is_file()


def test_missing_train_directory_denies_the_floor(tmp_path: Path) -> None:
    """Losing the train state denies the chain floor on the reconciliation route.

    The bootstrap marker is what raises a tier's durable chain floor to its
    bootstrap version. Delete the whole durable-train directory and nothing is
    left to raise it, so forward admission has to demand released train
    evidence for every slot from the adoption floor up to the live version.

    The refusal is asserted on ``reconcile_durable_change_train_startup`` --
    the route the daemon runs at startup (``polylogue/daemon/cli.py``) -- and
    not on ``initialize_active_archive_root``, because #5275 made archive
    bootstrap skip startup reconciliation for an archive that carries a valid
    ``.polylogue-format.json``. Opening is therefore the correct observable
    for bootstrap here; the chain-floor question is the reconciler's.

    Anti-vacuity: make ``_fresh_durable_bootstrap_versions`` fall back to
    ``ARCHIVE_VERSION_BY_TIER`` when the manifest root is missing and the
    floor is granted with no marker at all, so the refusal below disappears.
    ``test_pre_marker_current_archive_is_adopted_once`` pins the opposite
    direction: an archive that still has its train directory is re-adopted
    rather than refused.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    marker_root = tmp_path / ".maintenance-state" / "durable-change-trains"
    (marker_root / ".bootstrap").unlink()
    marker_root.rmdir()

    initialize_active_archive_root(tmp_path)
    assert not marker_root.exists()

    with pytest.raises(DurableChangeTrainError, match="lacks released train evidence"):
        reconcile_durable_change_train_startup(tmp_path)


def test_missing_durable_tier_is_never_recreated(tmp_path: Path) -> None:
    """A lineage member that lost a durable tier is refused, not re-bootstrapped.

    ``user.db`` is durable, irreplaceable state. An archive whose format
    marker names it must never have it silently recreated empty by the next
    open -- the refusal names the missing file and leaves the root alone.

    The refusal now comes from the archive format marker
    (``assert_archive_format_lineage``) rather than from durable-train
    admission, which is strictly earlier: it fires before any tier is opened.

    Anti-vacuity: drop the per-tier existence check from
    ``assert_archive_format_lineage`` and bootstrap recreates ``user.db`` as
    an empty canonical tier, so both the raise and the final assertion go red.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    (tmp_path / ".maintenance-state" / "durable-change-trains" / ".bootstrap").unlink()
    (tmp_path / "user.db").unlink()

    with pytest.raises(RuntimeError, match="marker names a missing durable tier"):
        initialize_active_archive_root(tmp_path)
    assert not (tmp_path / "user.db").exists()


def test_bootstrap_marker_survives_index_generation_replacement(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    (tmp_path / "index.db").unlink()

    initialize_active_archive_root(tmp_path)


def test_fresh_bootstrap_archive_opens_after_its_root_is_moved(tmp_path: Path) -> None:
    """The fresh-start ruling's rollback route -- move the files back -- must open.

    ``os.rename`` within one filesystem changes the archive root path and
    nothing else: every tier keeps its device and inode. An archive created by
    current code must still open afterwards, because that move is the whole
    safety net behind the rebuild campaign (polylogue-ifb4l).

    Anti-vacuity: restoring the committed marker's ``durable_identity_digest``
    (sha256 over the configured root path plus the source/user
    ``dev:<st_dev>:ino:<st_ino>`` pair) and comparing it in
    ``_fresh_durable_bootstrap_versions`` makes this raise
    ``fresh durable bootstrap marker durable identity mismatch``.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    origin = tmp_path / "origin"
    initialize_active_archive_root(origin)
    marker_payload = json.loads(
        (origin / ".maintenance-state" / "durable-change-trains" / ".bootstrap").read_text(encoding="utf-8")
    )
    assert "durable_identity_digest" not in marker_payload

    moved = tmp_path / "moved"
    os.rename(origin, moved)

    initialize_active_archive_root(moved)
    assert reconcile_durable_change_train_startup(moved) == ()


def test_fresh_bootstrap_marker_is_refused_in_an_archive_it_does_not_describe(tmp_path: Path) -> None:
    """One archive's bootstrap authority cannot be transplanted into another.

    This is the property the removed path/inode seal was protecting: the marker
    raises the durable train chain floor, so an archive that acquired a foreign
    marker would stop needing released train evidence for every version between
    the adoption floor and the recorded bootstrap version. The replacement proof
    is the recipient's own durable content -- its live tier must still be the
    canonical schema the marker describes, or a released train on that archive
    must have proved it was.

    Anti-vacuity: deleting the ``_assert_fresh_durable_bootstrap_is_own`` call
    from ``_fresh_durable_bootstrap_versions`` makes the recipient open and
    silently inherit the donor's chain floor.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    donor = tmp_path / "donor"
    recipient = tmp_path / "recipient"
    initialize_active_archive_root(donor)
    initialize_active_archive_root(recipient)

    relative = Path(".maintenance-state") / "durable-change-trains" / ".bootstrap"
    (recipient / relative).unlink()
    with closing(sqlite3.connect(recipient / "source.db")) as connection:
        connection.execute("CREATE INDEX idx_transplanted_marker_probe ON raw_sessions(raw_id)")
        connection.commit()
    shutil.copyfile(donor / relative, recipient / relative)
    with closing(sqlite3.connect(recipient / "source.db")) as connection:
        live = migration_runner.capture_durable_schema_inventory(connection)
    assert (
        live.sha256
        != durable_change_train_module._canonical_schema_inventory(
            ArchiveTier.SOURCE, ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
        ).sha256
    )

    with pytest.raises(DurableChangeTrainError, match="not this archive's own source bootstrap evidence"):
        reconcile_durable_change_train_startup(recipient)


#: The tier the skew tests below bend, and the tier they leave alone. A marker
#: only ever grants a tier something *above* its adoption floor, so the skew
#: branch is only observable on a tier whose target sits above that floor --
#: with the floor at 1, ``audit`` (target 1) is granted nothing either way and
#: would make the assertion pass for the wrong reason.
_SKEWABLE_TIER = ArchiveTier.SOURCE
_CORROBORATING_TIER = ArchiveTier.USER


def _skew_live_tier(archive_root: Path, tier: ArchiveTier) -> int:
    """Move one live tier off its declared version and return the new value."""
    declared = ARCHIVE_VERSION_BY_TIER[tier]
    assert declared > DURABLE_MIGRATION_ADOPTION_FLOORS[tier], (
        f"{tier.value} is at its adoption floor, so a marker grants it nothing and this fixture proves nothing"
    )
    skewed = declared - 1
    with closing(sqlite3.connect(archive_root / f"{tier.value}.db")) as connection:
        connection.execute(f"PRAGMA user_version = {skewed}")
        connection.commit()
    return skewed


def test_fresh_bootstrap_marker_grants_nothing_for_skew(tmp_path: Path) -> None:
    """A tier standing at a different version must park, not fail startup.

    A live tier whose own ``user_version`` disagrees with the marker is
    ordinary durable schema skew -- the condition polylogue-39pdi requires the
    daemon to survive in a degraded state. The marker simply grants that tier
    nothing; the tiers it still corroborates keep their authority.

    Anti-vacuity: removing the ``_fresh_durable_bootstrap_tier_version_skew``
    branch from ``_assert_fresh_durable_bootstrap_is_own`` makes this raise
    ``not this archive's own source bootstrap evidence`` instead of returning.
    Verified by reverting the branch, not by asserting it. The second
    assertion pins the opposite direction: a guard that simply granted nothing
    would drop the corroborating tier too.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    _skew_live_tier(tmp_path, _SKEWABLE_TIER)

    manifest_root = tmp_path / ".maintenance-state" / "durable-change-trains"
    granted = durable_change_train_module._fresh_durable_bootstrap_versions(tmp_path, manifest_root)

    assert _SKEWABLE_TIER not in granted
    assert granted[_CORROBORATING_TIER] == ARCHIVE_VERSION_BY_TIER[_CORROBORATING_TIER]


def test_legacy_sealed_marker_grants_nothing_for_skew(tmp_path: Path) -> None:
    """The legacy seal proves ownership, not that a skewed tier is current.

    Markers written by earlier revisions carry a path-and-inode
    ``durable_identity_digest``. Matching it establishes the marker is this
    archive's own -- so nothing is transplanted -- but a tier standing at a
    different ``user_version`` still grants nothing, exactly as it does for a
    marker carrying no seal.

    Anti-vacuity: restoring the unconditional ``return set()`` in the legacy
    branch makes this fail with the skewed tier present in the granted
    versions. Verified by reverting, not by asserting. The second assertion
    pins the opposite direction, where the seal stops granting anything at all.
    """
    from polylogue.storage.archive_identity import ArchiveIdentity
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    manifest_root = tmp_path / ".maintenance-state" / "durable-change-trains"
    marker = manifest_root / ".bootstrap"

    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["durable_identity_digest"] = durable_change_train_module._durable_identity_digest(
        ArchiveIdentity.resolve(tmp_path)
    )
    payload.pop("marker_digest", None)
    payload["marker_digest"] = durable_change_train_module._bootstrap_marker_digest(payload)
    marker.write_text(json.dumps(payload), encoding="utf-8")

    _skew_live_tier(tmp_path, _SKEWABLE_TIER)

    granted = durable_change_train_module._fresh_durable_bootstrap_versions(tmp_path, manifest_root)

    assert _SKEWABLE_TIER not in granted
    assert granted[_CORROBORATING_TIER] == ARCHIVE_VERSION_BY_TIER[_CORROBORATING_TIER]


def test_fresh_bootstrap_marker_is_retired_once_it_grants_nothing(tmp_path: Path) -> None:
    """The marker is removed as soon as it stops carrying authority.

    A fresh archive's marker records versions above every adoption floor, so it
    is the archive's only evidence for them and must stay. Once the recorded
    versions are covered without it -- here, versions at the adoption floors
    themselves -- keeping it would gate the archive on bootstrap evidence for
    the rest of its life for nothing, so startup deletes it.

    Anti-vacuity: the first half goes red if retirement stops consulting the
    chain requirement and deletes unconditionally; the second half goes red if
    the "grants nothing" branch is removed. Both verified by reverting.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.durable_change_train import (
        _retire_corroborated_fresh_durable_bootstrap_marker,
    )

    initialize_active_archive_root(tmp_path)
    manifest_root = tmp_path / ".maintenance-state" / "durable-change-trains"
    marker = manifest_root / ".bootstrap"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    recorded = {
        ArchiveTier[name.upper()]: version for name, version in cast(dict[str, int], payload["versions"]).items()
    }
    assert any(version > DURABLE_MIGRATION_ADOPTION_FLOORS[tier] for tier, version in recorded.items())

    assert _retire_corroborated_fresh_durable_bootstrap_marker(manifest_root, recorded) is False
    assert marker.is_file()
    assert reconcile_durable_change_train_startup(tmp_path) == ()
    assert marker.is_file()

    floors = {tier: DURABLE_MIGRATION_ADOPTION_FLOORS[tier] for tier in recorded}
    assert _retire_corroborated_fresh_durable_bootstrap_marker(manifest_root, floors) is True
    assert not marker.exists()


def test_fresh_bootstrap_receipt_rejects_recorded_version_tampering(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    marker = tmp_path / ".maintenance-state" / "durable-change-trains" / ".bootstrap"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    cast(dict[str, int], payload["versions"])["source"] += 1
    marker.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DurableChangeTrainError, match="marker digest mismatch"):
        reconcile_durable_change_train_startup(tmp_path)


def test_source_train_identity_survives_late_user_tier_initialization(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database

    source_path = tmp_path / "source.db"
    initialize_archive_database(source_path, ArchiveTier.SOURCE)
    with sqlite3.connect(source_path) as conn:
        before = migration_runner.capture_durable_database_evidence(conn, ArchiveTier.SOURCE)

    initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER)
    with sqlite3.connect(source_path) as conn:
        after = migration_runner.capture_durable_database_evidence(conn, ArchiveTier.SOURCE)

    assert after.archive_identity_digest == before.archive_identity_digest


def test_future_train_sidecar_hash_and_slot_are_admission_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "fixture_migrations_hash"
    source_package = package_root / "source"
    source_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (source_package / "__init__.py").write_text("", encoding="utf-8")
    sql = "-- migration-safety: additive-no-backup\nCREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;\n"
    sql_path = source_package / _NEXT_SOURCE_SQL_NAME
    sql_path.write_text(sql, encoding="utf-8")
    claim = durable_migration_claim_for_sql(ArchiveTier.SOURCE, sql_path.name, sql, owner_ref="owner:future-source")
    train = declare_durable_change_train(
        train_id=f"train:source:v{_NEXT_SOURCE_SLOT}",
        tier=ArchiveTier.SOURCE,
        current_version=_SOURCE_ADOPTION_FLOOR,
        target_version=_NEXT_SOURCE_SLOT,
        slot=_NEXT_SOURCE_SLOT,
        owner_ref="owner:future-source",
        migration=claim,
        riders=(_rider(),),
        declared_at_ms=1,
    )
    payload = migration_runner.durable_change_train_to_payload(train)
    cast(dict[str, object], payload["migration"])["sql_sha256"] = "0" * 64
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = migration_runner._canonical_json_sha256(payload)
    (source_package / _NEXT_SOURCE_SIDECAR_NAME).write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda _tier: "fixture_migrations_hash.source",
    )

    with pytest.raises(DurableChangeTrainError, match="SQL SHA-256 mismatch"):
        validate_durable_migration_sidecars(ArchiveTier.SOURCE, ((sql_path.name, sql),))

    cast(dict[str, object], payload["migration"])["sql_sha256"] = claim.sql_sha256
    payload["slot"] = _NEXT_SOURCE_SLOT + 1
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = migration_runner._canonical_json_sha256(payload)
    (source_package / _NEXT_SOURCE_SIDECAR_NAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DurableChangeTrainError, match="slot"):
        validate_durable_migration_sidecars(ArchiveTier.SOURCE, ((sql_path.name, sql),))


def test_missing_future_sidecar_is_rejected_at_the_migration_runner_choke_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "fixture_migrations_missing"
    source_package = package_root / "source"
    source_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (source_package / "__init__.py").write_text("", encoding="utf-8")
    sql = "-- migration-safety: additive-no-backup\nCREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;\n"
    sql_path = source_package / _NEXT_SOURCE_SQL_NAME
    sql_path.write_text(sql, encoding="utf-8")
    claim = durable_migration_claim_for_sql(ArchiveTier.SOURCE, sql_path.name, sql, owner_ref="owner:future-source")
    train = declare_durable_change_train(
        train_id=f"train:source:v{_NEXT_SOURCE_SLOT}",
        tier=ArchiveTier.SOURCE,
        current_version=_SOURCE_ADOPTION_FLOOR,
        target_version=_NEXT_SOURCE_SLOT,
        slot=_NEXT_SOURCE_SLOT,
        owner_ref="owner:future-source",
        migration=claim,
        riders=(_rider(),),
        declared_at_ms=1,
    )
    sidecar = source_package / _NEXT_SOURCE_SIDECAR_NAME
    sidecar.write_text(json.dumps(migration_runner.durable_change_train_to_payload(train)), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(migration_runner, "_migration_package", lambda _tier: "fixture_migrations_missing.source")
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda _tier: "fixture_migrations_missing.source",
    )
    assert migration_runner._load_migrations(ArchiveTier.SOURCE)[0].version == _NEXT_SOURCE_SLOT
    sidecar.unlink()
    with pytest.raises(MigrationError, match="missing durable migration train sidecar"):
        migration_runner._load_migrations(ArchiveTier.SOURCE)
    policy = durable_change_train_policy_report(ArchiveTier.SOURCE)
    assert policy["ok"] is False
    violations = cast(list[str], policy["violations"])
    assert any("missing durable migration train sidecar" in violation for violation in violations)
    (source_package / f"{_NEXT_SOURCE_SLOT + 1:03d}.train.json").write_text(
        json.dumps(migration_runner.durable_change_train_to_payload(train)), encoding="utf-8"
    )
    with pytest.raises(DurableChangeTrainError, match="no matching SQL resource"):
        validate_durable_migration_sidecars(ArchiveTier.SOURCE, ((sql_path.name, sql),))


_TRANSACTION_ESCAPE_SQL = "CREATE TABLE transaction_escape (id INTEGER PRIMARY KEY);\nCOMMIT;\n"


def test_migration_sql_cannot_escape_the_transaction(tmp_path: Path) -> None:
    """A migration statement may not end the runner's own transaction.

    The runner wraps every numbered slot in one ``BEGIN IMMEDIATE`` so a
    mid-file failure rolls the durable tier back to its current version. A
    ``COMMIT`` inside the SQL would end that transaction early and make the
    preceding statements unrecoverable, so the executor refuses transaction
    control at the statement it reads.

    This drives ``_execute_migration_sql`` -- the executor production calls
    from inside its lock -- rather than ``migrate_archive_tier``, because an
    ``additive-no-backup`` file carrying ``COMMIT`` is now refused earlier
    still, at discovery (``test_additive_claim_refusal_precedes_writes``
    below). The transaction-control refusal remains the live guard for a
    backup-requiring slot, which carries no additive marker and so reaches
    the executor.

    Anti-vacuity: delete the ``_SQL_TRANSACTION_CONTROL_RE`` refusal in
    ``_execute_migration_sql`` and the ``COMMIT`` runs, the ``pytest.raises``
    reports DID NOT RAISE, and the rollback below no longer removes
    ``transaction_escape`` because it was already committed.
    """
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(MigrationError, match="must not control the existing transaction"):
            migration_runner._execute_migration_sql(conn, _TRANSACTION_ESCAPE_SQL)
        conn.rollback()
        assert conn.execute("PRAGMA user_version").fetchone() == (_CURRENT_VERSION,)
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='transaction_escape'").fetchone() is None


def test_additive_claim_refusal_precedes_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A false ``additive-no-backup`` claim is refused before the tier is touched.

    Slot discovery proves the marker against the file's own statements, and
    ``COMMIT`` is not an additive statement. The refusal has to land before
    the migration lock, or the durable tier would already have been written
    by the time the contradiction is noticed.

    Anti-vacuity: drop the ``_assert_additive_migration_sql`` call from
    ``_requires_migration_backup`` and this file is accepted as backup-waived,
    so the observable error changes from the additive refusal to the
    transaction-control one -- and the bytes assertion is what proves the
    refusal preceded any write rather than following a partial apply.
    """
    package_name = "fixture_migrations_false_additive"
    tier_package = tmp_path / package_name / ArchiveTier.SOURCE.value
    tier_package.mkdir(parents=True)
    (tmp_path / package_name / "__init__.py").write_text("", encoding="utf-8")
    (tier_package / "__init__.py").write_text("", encoding="utf-8")
    (tier_package / "002_durable_items.sql").write_text(
        f"-- migration-safety: additive-no-backup\n{_TRANSACTION_ESCAPE_SQL}", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[ArchiveTier.SOURCE] = _TARGET_VERSION
    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)
    monkeypatch.setattr(
        migration_runner, "_migration_package", lambda observed_tier: f"{package_name}.{observed_tier.value}"
    )
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda observed_tier: f"{package_name}.{observed_tier.value}",
    )

    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    before = db_path.read_bytes()

    with sqlite3.connect(db_path) as conn:
        with pytest.raises(MigrationError, match="not additive-only"):
            migration_runner.migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=None)
        assert conn.execute("PRAGMA user_version").fetchone() == (_CURRENT_VERSION,)
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='transaction_escape'").fetchone() is None
    assert db_path.read_bytes() == before


def test_canonical_inventory_preserves_trigger_literal_whitespace() -> None:
    def inventory(trigger_literal: str, *, formatted: bool) -> str:
        with sqlite3.connect(":memory:") as conn:
            conn.executescript(
                """
                CREATE TABLE items (item_id INTEGER PRIMARY KEY);
                CREATE TABLE audit (payload TEXT NOT NULL);
                """
            )
            if formatted:
                conn.executescript(
                    f"""
                    CREATE TRIGGER audit_items
                    AFTER INSERT ON items
                    BEGIN
                        INSERT INTO audit(payload) VALUES ({trigger_literal});
                    END;
                    """
                )
            else:
                conn.executescript(
                    "CREATE TRIGGER audit_items AFTER INSERT ON items BEGIN "
                    f"INSERT INTO audit(payload) VALUES({trigger_literal}); END;"
                )
            return migration_runner.capture_durable_schema_inventory(conn).sha256

    spaced = inventory("'a  b'", formatted=True)
    same_literal_compact_layout = inventory("'a  b'", formatted=False)
    changed_literal = inventory("'a b'", formatted=False)

    assert spaced == same_literal_compact_layout
    assert spaced != changed_literal


def _source_inventory_refs_at(target: int) -> set[str]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(ARCHIVE_DDL_BY_TIER[ArchiveTier.SOURCE])
        migration_runner._prepare_fresh_connection_for_target(connection, ArchiveTier.SOURCE, target)
        inventory = migration_runner.capture_durable_schema_inventory(connection)
    finally:
        connection.close()
    return {item.object_ref for item in inventory.objects}


def test_source_inventory_projects_away_future_objects() -> None:
    """Historical parity keeps objects at the target and removes later additions.

    The shipped bootstrap DDL is always the newest shape, so proving parity
    for an archive paused *below* the current target means projecting the
    canonical schema back to that slot. The objects that must disappear are
    exactly the ones later slots' riders declare -- here the source tier's one
    numbered slot (002) and its ``excision_policy_projections`` carrier.

    Anti-vacuity: drop the ``future_refs`` removal loop in
    ``_prepare_fresh_connection_for_target`` and the projection keeps
    ``table:excision_policy_projections``, so the floor assertion goes red
    while the target assertion stays green. The ``at_target`` case pins the
    opposite direction: a projection that dropped the object unconditionally
    -- or at the tier's own target -- would be red there.
    """
    floor = _SOURCE_ADOPTION_FLOOR
    at_target = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
    assert at_target > floor, "the source tier owns no numbered slot, so nothing can be projected away"

    projected = _source_inventory_refs_at(floor)
    current = _source_inventory_refs_at(at_target)

    assert "table:source_items" in projected
    assert "table:material_observations" in projected
    assert "table:excision_policy_projections" not in projected
    assert "table:excision_policy_projections" in current


def test_admission_rejects_stale_current_and_target_versions() -> None:
    train = _declared(ArchiveTier.SOURCE)
    with pytest.raises(DurableChangeTrainError, match="stale durable train current"):
        admit_durable_change_train(
            train,
            observed_current_version=0,
            fresh_ddl_parity=_parity(ArchiveTier.SOURCE),
            admission_evidence_ref="proof:admit",
            migration_claims=(train.migration,),
            canonical_target_version=_TARGET_VERSION,
        )
    with pytest.raises(DurableChangeTrainError, match="stale durable train target"):
        admit_durable_change_train(
            train,
            observed_current_version=_CURRENT_VERSION,
            fresh_ddl_parity=_parity(ArchiveTier.SOURCE),
            admission_evidence_ref="proof:admit",
            migration_claims=(train.migration,),
            canonical_target_version=_TARGET_VERSION + 1,
        )


def test_slot_collision_names_both_owners_and_blocks(tmp_path: Path) -> None:
    """Two files claiming one slot are named as a collision and refused admission.

    Contention is on ``(tier, target version, slot)``, so a late rider that
    renumbers itself onto an already-owned slot has to be reported with *both*
    owners -- the operator's fix is rebase/renumber, which needs to know what
    it is colliding with.

    The slot is built from synthetic SQL rather than shipped migration files.
    The pre-reset version of this test read ``008_raw_session_capture_mode.sql``
    and ``009_expand_origin_vocabulary.sql`` off disk; the lineage reset
    deleted both, and the property under test never depended on their content.

    Anti-vacuity: make ``find_durable_migration_collisions`` return ``()`` and
    the report goes ``ok: True`` while ``admit_durable_change_train`` accepts
    both claims, so every assertion below goes red.
    """
    slot = _NEXT_SOURCE_SLOT
    first_name = f"{slot:03d}_first_items.sql"
    late_name = f"{slot:03d}_late_items.sql"
    first = durable_migration_claim_for_sql(
        ArchiveTier.SOURCE,
        first_name,
        "-- migration-safety: additive-no-backup\nCREATE TABLE first_items (id INTEGER PRIMARY KEY) STRICT;\n",
        owner_ref="owner:source-first",
    )
    late_rider = durable_migration_claim_for_sql(
        ArchiveTier.SOURCE,
        late_name,
        "-- migration-safety: additive-no-backup\nCREATE TABLE late_items (id INTEGER PRIMARY KEY) STRICT;\n",
        owner_ref="owner:source-late-rider",
    )
    report = durable_migration_collision_report((first, late_rider))
    assert report["ok"] is False
    serialized = json.dumps(report)
    assert first_name in serialized
    assert late_name in serialized
    assert "owner:source-first" in serialized
    assert "owner:source-late-rider" in serialized

    train = declare_durable_change_train(
        train_id=f"train:source:v{slot}",
        tier=ArchiveTier.SOURCE,
        current_version=slot - 1,
        target_version=slot,
        slot=slot,
        owner_ref="owner:source-train",
        migration=first,
        riders=(_rider(),),
    )
    parity = DurableFreshDDLParityProof(
        tier=ArchiveTier.SOURCE,
        target_version=slot,
        migrated_version=slot,
        fresh_version=slot,
        migrated_inventory_sha256="a" * 64,
        parity_inventory_sha256="a" * 64,
        fresh_inventory_sha256="a" * 64,
        missing_objects=(),
        unexpected_objects=(),
        changed_objects=(),
        evidence_ref=f"proof:v{slot}-fresh",
        matches=True,
    )
    with pytest.raises(DurableChangeTrainError, match="collision.*rebase/renumber") as exc_info:
        admit_durable_change_train(
            train,
            observed_current_version=slot - 1,
            fresh_ddl_parity=parity,
            admission_evidence_ref=f"proof:v{slot}-admit",
            migration_claims=(first, late_rider),
            canonical_target_version=slot,
        )
    assert first_name in str(exc_info.value)
    assert late_name in str(exc_info.value)


def test_duplicate_train_ownership_and_late_rider_are_rejected() -> None:
    admitted = _admitted(ArchiveTier.SOURCE)
    duplicate = replace(_declared(ArchiveTier.SOURCE), train_id="train:source:v2:duplicate")
    with pytest.raises(DurableChangeTrainError, match="contention key already owned"):
        admit_durable_change_train(
            duplicate,
            observed_current_version=_CURRENT_VERSION,
            fresh_ddl_parity=_parity(ArchiveTier.SOURCE),
            admission_evidence_ref="proof:duplicate",
            active_trains=(admitted,),
            migration_claims=(duplicate.migration,),
            canonical_target_version=_TARGET_VERSION,
        )
    with pytest.raises(DurableChangeTrainError, match="late rider.*target v3"):
        add_durable_change_train_rider(admitted, _rider(trust_floor_exception_ref="exception:late"))


def test_schema_only_unproven_and_nonproduction_riders_fail_admission() -> None:
    schema_only = _rider(consumer_count=0, trust_floor_exception_ref="exception:single-consumer-floor")
    with pytest.raises(DurableChangeTrainError, match="schema-only"):
        _admitted(ArchiveTier.SOURCE, rider=schema_only)

    one_consumer = _rider(consumer_count=1)
    with pytest.raises(DurableChangeTrainError, match="fewer than two"):
        _admitted(ArchiveTier.SOURCE, rider=one_consumer)

    test_only = DurableChangeRider(
        rider_id="test-only",
        owner_ref="owner:test-only",
        schema_objects=("table:durable_items",),
        runtime_consumers=(
            DurableRuntimeConsumer("test-a", "tests/unit/test_a.py:test_a", "proof:test-a", ("read",)),
            DurableRuntimeConsumer("test-b", "fixture:test-b", "proof:test-b", ("write",)),
        ),
        behavior_proof_refs=("proof:test-a", "proof:test-b"),
    )
    with pytest.raises(DurableChangeTrainError, match="test-only"):
        _admitted(ArchiveTier.SOURCE, rider=test_only)


def test_fresh_ddl_parity_mismatch_blocks_admission() -> None:
    mismatch = _parity(ArchiveTier.SOURCE, include_durable_items=False)
    assert mismatch.matches is False
    train = _declared(ArchiveTier.SOURCE)
    with pytest.raises(DurableChangeTrainError, match="fresh-DDL parity"):
        admit_durable_change_train(
            train,
            observed_current_version=_CURRENT_VERSION,
            fresh_ddl_parity=mismatch,
            admission_evidence_ref="proof:mismatch",
            migration_claims=(train.migration,),
            canonical_target_version=_TARGET_VERSION,
        )


def test_backup_authority_timestamp_precedes_its_captured_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authorization must not depend on two clock reads landing in one millisecond."""
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    train = _admitted(ArchiveTier.SOURCE)
    train = reserve_durable_change_train(
        train,
        reservation_id="lease:source",
        reservation_owner_ref=train.owner_ref,
        archive_root=tmp_path,
        tier_path=db_path,
        daemon_stopped_evidence_ref="proof:stopped",
        single_writer_evidence_ref="proof:lease",
        reserved_at_ms=3,
    )
    timestamps = iter((10, 11))
    monkeypatch.setattr(migration_runner, "_durable_now_ms", lambda: next(timestamps))

    with sqlite3.connect(db_path) as conn:
        authorized = authorize_durable_change_train_backup(
            conn,
            train,
            backup_manifest=None,
            evidence_ref="proof:additive-no-backup",
        )

    assert authorized.backup_authorization is not None
    assert authorized.pre_apply_evidence is not None
    assert authorized.backup_authorization.authorized_at_ms == 10
    assert authorized.pre_apply_evidence.observed_at_ms == 11


def test_missing_backup_authority_stops_a_backup_required_train(tmp_path: Path) -> None:
    backup_sql = "CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;\n"
    claim = _claim(ArchiveTier.USER, backup_sql)
    train = _admitted(
        ArchiveTier.USER,
        claim=claim,
        backup_plan_ref="backup-profile:user-overlays",
    )
    db_path = tmp_path / "user.db"
    _create_current_database(db_path)
    train = reserve_durable_change_train(
        train,
        reservation_id="lease:user",
        reservation_owner_ref=train.owner_ref,
        archive_root=tmp_path,
        tier_path=db_path,
        daemon_stopped_evidence_ref="proof:stopped",
        single_writer_evidence_ref="proof:lease",
    )
    with sqlite3.connect(db_path) as conn:
        with pytest.raises(DurableChangeTrainError, match="requires an authenticated backup"):
            authorize_durable_change_train_backup(
                conn,
                train,
                backup_manifest=None,
                evidence_ref="proof:missing-backup",
            )


def test_source_and_user_share_only_the_same_archive_writer_reservation(tmp_path: Path) -> None:
    source = _admitted(ArchiveTier.SOURCE, owner_ref="owner:operator")
    user = _admitted(ArchiveTier.USER, owner_ref="owner:operator")
    source_reserved = reserve_durable_change_train(
        source,
        reservation_id="lease:shared",
        reservation_owner_ref="owner:operator",
        archive_root=tmp_path,
        tier_path=tmp_path / "source.db",
        daemon_stopped_evidence_ref="proof:stopped",
        single_writer_evidence_ref="proof:lease",
    )
    with pytest.raises(DurableChangeTrainError, match="second writer rejected"):
        reserve_durable_change_train(
            user,
            reservation_id="lease:other",
            reservation_owner_ref="owner:operator",
            archive_root=tmp_path,
            tier_path=tmp_path / "user.db",
            daemon_stopped_evidence_ref="proof:stopped",
            single_writer_evidence_ref="proof:other-lease",
            active_trains=(source_reserved,),
        )
    user_reserved = reserve_durable_change_train(
        user,
        reservation_id="lease:shared",
        reservation_owner_ref="owner:operator",
        archive_root=tmp_path,
        tier_path=tmp_path / "user.db",
        daemon_stopped_evidence_ref="proof:stopped",
        single_writer_evidence_ref="proof:lease",
        active_trains=(source_reserved,),
    )
    assert source_reserved.reservation is not None
    assert user_reserved.reservation is not None
    assert user_reserved.reservation.reservation_id == source_reserved.reservation.reservation_id


def test_failed_transaction_exposes_exact_retry_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two conflicting CREATEs: the second raises "table durable_items already
    # exists" mid-apply, which is the transaction failure these tests exercise.
    # An `INSERT` here would make the file's own statement set contradict its
    # `additive-no-backup` header, which the runner now refuses at discovery.
    failing_sql = """-- migration-safety: additive-no-backup
CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;
CREATE TABLE durable_items (item_id TEXT PRIMARY KEY) STRICT;
"""
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    claim = _claim(ArchiveTier.SOURCE, failing_sql)
    train = _admitted(ArchiveTier.SOURCE, claim=claim)
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE, sql=failing_sql)
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        with pytest.raises(DurableChangeTrainApplyError) as exc_info:
            apply_durable_change_train(conn, train)
        failed = exc_info.value.failed_train
        assert failed.state is DurableChangeTrainState.FAILED
        assert failed.failure is not None
        assert failed.failure.classification is DurableFailureClassification.ROLLED_BACK_TO_CURRENT
        failure_manifest = tmp_path / "source-failed-train.json"
        write_durable_change_train_manifest(failure_manifest, failed, expected_revision=-1)
        failed = load_durable_change_train_manifest(failure_manifest)
        released_failed = record_durable_writer_release(failed, evidence_ref="proof:failed-writer-release")
        released_manifest = tmp_path / ".maintenance-state" / "durable-change-trains" / "source-002.json"
        released_manifest.parent.mkdir(parents=True)
        write_durable_change_train_manifest(released_manifest, released_failed, expected_revision=-1)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == _CURRENT_VERSION
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='durable_items'").fetchone() is None
        recovered = recover_durable_change_train(
            conn,
            failed,
            recovery_evidence_ref="proof:rollback-observed",
            writer_release_evidence_ref="proof:lease-released",
        )
    assert recovered.state is DurableChangeTrainState.ADMITTED
    assert recovered.reservation is None
    assert recovered.backup_authorization is None
    assert recovered.pre_apply_evidence is None


def test_interrupted_commit_recovers_at_applied_without_reapplying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    train = _admitted(ArchiveTier.SOURCE)
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        conn.execute("CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT")
        conn.execute(f"PRAGMA user_version = {_TARGET_VERSION}")
        conn.commit()

        def must_not_reapply(*_args: object, **_kwargs: object) -> None:
            pytest.fail("interrupted target recovery re-entered the migration engine")

        monkeypatch.setattr(migration_runner, "migrate_archive_tier", must_not_reapply)
        recovered = reconcile_interrupted_durable_change_train(
            conn,
            train,
            interruption_evidence_ref="proof:process-died-after-commit",
            writer_release_evidence_ref="proof:lease-expired",
        )
        recovered_manifest = tmp_path / "source-interrupted-recovered.json"
        write_durable_change_train_manifest(recovered_manifest, recovered, expected_revision=-1)
        recovered = load_durable_change_train_manifest(recovered_manifest)
    assert recovered.state is DurableChangeTrainState.APPLIED
    assert recovered.apply_evidence is not None
    assert recovered.apply_evidence.recovered_after_interrupt is True
    assert recovered.reservation is not None and recovered.reservation.active is False


def test_startup_reconciles_interrupted_train_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted mid-apply train is finished and persisted at startup.

    The train is BACKUP_AUTHORIZED with a durable file that already carries
    the committed slot: the crash landed between the SQL commit and the
    manifest transition. Startup reconciliation has to recognise that, record
    ``recovered_after_interrupt``, and release -- not re-apply.

    The refusal is driven through ``reconcile_durable_change_trains_on_startup``
    rather than ``initialize_active_archive_root``. #5275 made archive
    bootstrap skip startup reconciliation whenever a valid
    ``.polylogue-format.json`` is present, and the daemon
    (``polylogue/daemon/cli.py``) calls the reconciler itself; a hand-authored
    fixture archive carries no format marker and would be refused for that
    alone, which is not the behaviour under test.

    Anti-vacuity: make ``reconcile_interrupted_durable_change_train`` return
    the train unchanged and the state stays BACKUP_AUTHORIZED with no apply
    evidence, so every assertion below goes red.
    """
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER, bootstrap

    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[ArchiveTier.SOURCE] = _TARGET_VERSION
    monkeypatch.setattr(bootstrap, "ARCHIVE_VERSION_BY_TIER", versions)
    ddl = dict(ARCHIVE_DDL_BY_TIER)
    ddl[ArchiveTier.SOURCE] = (
        "CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT; "
        "CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;"
    )
    monkeypatch.setattr(bootstrap, "ARCHIVE_DDL_BY_TIER", ddl)
    # Fresh-DDL parity projects the runtime target back to the train's slot.
    # Left unpatched, the runner replays the real source migration history over
    # this synthetic tier and fails on tables the fixture never declares.
    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    train = _admitted(ArchiveTier.SOURCE, rider=_production_rider())
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        conn.execute("CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT")
        conn.execute(f"PRAGMA user_version = {_TARGET_VERSION}")
        conn.commit()
    manifest = tmp_path / ".maintenance-state" / "durable-change-trains" / "source-002.json"
    write_durable_change_train_manifest(manifest, train, expected_revision=-1)

    from polylogue.operations.durable_change_train import reconcile_durable_change_trains_on_startup

    assert reconcile_durable_change_trains_on_startup(tmp_path) == (manifest,)
    recovered = load_durable_change_train_manifest(manifest)
    assert recovered.state is DurableChangeTrainState.RELEASED
    assert recovered.apply_evidence is not None
    assert recovered.apply_evidence.recovered_after_interrupt is True
    assert "proof:startup-recovery:train:source:v2" in recovered.proof_refs


def test_bootstrap_finishes_persisted_applied_train_without_reapplying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER, bootstrap

    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[ArchiveTier.SOURCE] = _TARGET_VERSION
    monkeypatch.setattr(bootstrap, "ARCHIVE_VERSION_BY_TIER", versions)
    ddl = dict(ARCHIVE_DDL_BY_TIER)
    ddl[ArchiveTier.SOURCE] = (
        "CREATE TABLE base_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT; "
        "CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;"
    )
    monkeypatch.setattr(bootstrap, "ARCHIVE_DDL_BY_TIER", ddl)
    # Fresh-DDL parity projects the runtime target back to the train's slot.
    # Left unpatched, the runner replays the real source migration history over
    # this synthetic tier and fails on tables the fixture never declares.
    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    train = _admitted(ArchiveTier.SOURCE, rider=_production_rider())
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        conn.execute("CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT")
        conn.execute(f"PRAGMA user_version = {_TARGET_VERSION}")
        conn.commit()
        train = reconcile_interrupted_durable_change_train(
            conn,
            train,
            interruption_evidence_ref="proof:post-commit-crash",
            writer_release_evidence_ref="proof:lease-expired",
        )
    assert train.reservation is not None
    train = replace(
        train,
        revision=train.revision + 1,
        reservation=replace(train.reservation, active=True, released_at_ms=None, release_evidence_ref=None),
    )
    manifest = tmp_path / ".maintenance-state" / "durable-change-trains" / "source-002.json"
    write_durable_change_train_manifest(manifest, train, expected_revision=-1)
    monkeypatch.setattr(
        migration_runner,
        "migrate_archive_tier",
        lambda *_args, **_kwargs: pytest.fail("startup attempted to reapply a committed train"),
    )

    from polylogue.storage.sqlite.archive_tiers.bootstrap import reconcile_durable_change_trains_on_startup

    assert reconcile_durable_change_trains_on_startup(tmp_path) == (manifest,)
    recovered = load_durable_change_train_manifest(manifest)
    assert recovered.state is DurableChangeTrainState.RELEASED
    assert recovered.proof is not None
    assert recovered.apply_evidence is not None and recovered.apply_evidence.recovered_after_interrupt is True


@pytest.mark.parametrize("tier", (ArchiveTier.SOURCE, ArchiveTier.USER))
@pytest.mark.parametrize("replacement", ("missing", "content", "inode"))
def test_startup_proves_durable_continuity_before_initialization_or_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier: ArchiveTier,
    replacement: str,
) -> None:
    """A lost or replaced durable file cannot be released over an APPLIED train.

    An APPLIED train holds an active writer reservation and pre/post evidence
    for one exact file. If that file is lost, edited, or swapped for another
    inode, reconciliation must refuse and leave the manifest APPLIED with its
    reservation held -- never initialize a replacement tier over it, and never
    release the train as if the migration it describes were still proven.

    The refusal is driven through ``reconcile_durable_change_trains_on_startup``
    rather than ``initialize_active_archive_root``. #5275 made archive
    bootstrap skip startup reconciliation for any archive carrying a valid
    ``.polylogue-format.json``, so the reconciler -- the route the daemon runs
    at startup -- is where this continuity proof now lives.

    Anti-vacuity: drop the live-tier evidence comparison from
    ``_verify_released_train_live_tier``/the APPLIED reconciliation branch and
    every parameter goes green with the manifest advanced past APPLIED.
    """
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    db_path = tmp_path / f"{tier.value}.db"
    _create_current_database(db_path)
    bootstrap.initialize_archive_database(tmp_path / "audit.db", ArchiveTier.AUDIT)
    _install_synthetic_migration(tmp_path, monkeypatch, tier)
    train = _admitted(tier, rider=_production_rider())
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        train = apply_durable_change_train(conn, train)
    manifest = tmp_path / ".maintenance-state" / "durable-change-trains" / f"{tier.value}-002.json"
    write_durable_change_train_manifest(manifest, train, expected_revision=-1)

    if replacement == "missing":
        db_path.unlink()
    elif replacement == "content":
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE base_items SET payload = 'lost' WHERE item_id = 'base-1'")
            conn.commit()
    else:
        replacement_path = tmp_path / "replacement.db"
        replacement_path.write_bytes(db_path.read_bytes())
        os.replace(replacement_path, db_path)

    from polylogue.operations.durable_change_train import reconcile_durable_change_trains_on_startup

    with pytest.raises(DurableChangeTrainError, match="refusing startup initialization/release"):
        reconcile_durable_change_trains_on_startup(tmp_path)

    recovered = load_durable_change_train_manifest(manifest)
    assert recovered.state is DurableChangeTrainState.APPLIED
    assert recovered.reservation is not None and recovered.reservation.active is True
    if replacement == "missing":
        assert not db_path.exists()


def test_startup_recovers_persisted_rollback_failure_to_admitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two conflicting CREATEs: the second raises "table durable_items already
    # exists" mid-apply, which is the transaction failure these tests exercise.
    # An `INSERT` here would make the file's own statement set contradict its
    # `additive-no-backup` header, which the runner now refuses at discovery.
    failing_sql = """-- migration-safety: additive-no-backup
CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;
CREATE TABLE durable_items (item_id TEXT PRIMARY KEY) STRICT;
"""
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE, sql=failing_sql)
    train = _admitted(ArchiveTier.SOURCE, claim=_claim(ArchiveTier.SOURCE, failing_sql))
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        with pytest.raises(DurableChangeTrainApplyError) as exc_info:
            apply_durable_change_train(conn, train)
        failed = exc_info.value.failed_train
    manifest = tmp_path / ".maintenance-state" / "durable-change-trains" / "source-002.json"
    write_durable_change_train_manifest(manifest, failed, expected_revision=-1)

    from polylogue.storage.sqlite.archive_tiers.bootstrap import reconcile_durable_change_trains_on_startup

    assert reconcile_durable_change_trains_on_startup(tmp_path) == (manifest,)
    recovered = load_durable_change_train_manifest(manifest)
    assert recovered.state is DurableChangeTrainState.ADMITTED
    assert recovered.reservation is None
    assert recovered.failure is None


@pytest.mark.parametrize("replacement", ("content", "inode"))
def test_startup_blocks_persisted_rollback_failure_after_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    # Two conflicting CREATEs: the second raises "table durable_items already
    # exists" mid-apply, which is the transaction failure these tests exercise.
    # An `INSERT` here would make the file's own statement set contradict its
    # `additive-no-backup` header, which the runner now refuses at discovery.
    failing_sql = """-- migration-safety: additive-no-backup
CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;
CREATE TABLE durable_items (item_id TEXT PRIMARY KEY) STRICT;
"""
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE, sql=failing_sql)
    train = _admitted(ArchiveTier.SOURCE, claim=_claim(ArchiveTier.SOURCE, failing_sql))
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        with pytest.raises(DurableChangeTrainApplyError) as exc_info:
            apply_durable_change_train(conn, train)
        failed = exc_info.value.failed_train
    manifest = tmp_path / ".maintenance-state" / "durable-change-trains" / "source-002.json"
    write_durable_change_train_manifest(manifest, failed, expected_revision=-1)

    if replacement == "content":
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE base_items SET payload = 'replaced' WHERE item_id = 'base-1'")
            conn.commit()
    else:
        replacement_path = tmp_path / "replacement.db"
        replacement_path.write_bytes(db_path.read_bytes())
        os.replace(replacement_path, db_path)

    from polylogue.storage.sqlite.archive_tiers.bootstrap import reconcile_durable_change_trains_on_startup

    with pytest.raises(DurableChangeTrainError, match="rolled-back recovery durable tier identity/content continuity"):
        reconcile_durable_change_trains_on_startup(tmp_path)
    blocked = load_durable_change_train_manifest(manifest)
    assert blocked.state is DurableChangeTrainState.FAILED
    assert blocked.reservation is not None and blocked.reservation.active is True


def test_startup_keeps_persisted_indeterminate_failure_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    train = _admitted(ArchiveTier.SOURCE)
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        conn.execute("PRAGMA user_version = 99")
        conn.commit()
        with pytest.raises(DurableChangeTrainRecoveryError) as exc_info:
            reconcile_interrupted_durable_change_train(
                conn,
                train,
                interruption_evidence_ref="proof:unknown-version",
                writer_release_evidence_ref="proof:lease-expired",
            )
        failed = exc_info.value.failed_train
    manifest = tmp_path / ".maintenance-state" / "durable-change-trains" / "source-002.json"
    write_durable_change_train_manifest(manifest, failed, expected_revision=-1)

    from polylogue.storage.sqlite.archive_tiers.bootstrap import reconcile_durable_change_trains_on_startup

    with pytest.raises(DurableChangeTrainRecoveryError, match="cannot recover automatically"):
        reconcile_durable_change_trains_on_startup(tmp_path)
    blocked = load_durable_change_train_manifest(manifest)
    assert blocked.state is DurableChangeTrainState.FAILED
    assert blocked.reservation is not None and blocked.reservation.active is True


def test_interrupted_unknown_version_requires_authenticated_restore(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    train = _admitted(ArchiveTier.SOURCE)
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        conn.execute("PRAGMA user_version = 99")
        conn.commit()
        with pytest.raises(
            DurableChangeTrainRecoveryError,
            match="restore the exact authenticated backup",
        ) as exc_info:
            reconcile_interrupted_durable_change_train(
                conn,
                train,
                interruption_evidence_ref="proof:unknown-version",
                writer_release_evidence_ref="proof:lease-expired",
            )
        failed = exc_info.value.failed_train
        assert failed.state is DurableChangeTrainState.FAILED
        assert failed.failure is not None
        assert failed.failure.classification is DurableFailureClassification.INDETERMINATE
        failure_manifest = tmp_path / "source-indeterminate-train.json"
        write_durable_change_train_manifest(failure_manifest, failed, expected_revision=-1)
        failed = load_durable_change_train_manifest(failure_manifest)
        assert failed.reservation is not None and failed.reservation.active is True
        assert failed.failure is not None
        assert "keep the daemon stopped" in failed.failure.required_actions
        with pytest.raises(DurableChangeTrainRecoveryError, match="retain stopped-daemon"):
            record_durable_writer_release(failed, evidence_ref="proof:unsafe-release")


def test_restart_and_every_runtime_consumer_are_required_before_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "user.db"
    _create_current_database(db_path)
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.USER)
    train = _admitted(ArchiveTier.USER)
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        train = apply_durable_change_train(conn, train)
    train = record_durable_writer_release(train, evidence_ref="proof:lease-release")
    with sqlite3.connect(db_path) as restarted:
        with _memory_target() as fresh:
            actual_parity = prove_durable_fresh_ddl_parity(
                ArchiveTier.USER,
                _TARGET_VERSION,
                migrated_connection=restarted,
                fresh_connection=fresh,
                evidence_ref="proof:actual-fresh",
            )
        incomplete = (DurableRuntimeConsumerResult("consumer-0", "proof:behavior:0", True),)
        restart = capture_durable_restart_convergence(
            restarted,
            train,
            runtime_consumers=incomplete,
            evidence_ref="proof:incomplete-restart",
        )
    assert restart.converged is False
    with pytest.raises(DurableChangeTrainError, match="runtime proof does not cover"):
        prove_durable_change_train(
            train,
            fresh_ddl_parity=actual_parity,
            runtime_consumers=incomplete,
            restart_convergence=restart,
        )
    with pytest.raises(DurableChangeTrainError, match="only a proven train"):
        release_durable_change_train(train, evidence_ref="proof:premature-release")


def test_manifest_semantics_reject_out_of_order_lifecycle_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE)
    train = _admitted(ArchiveTier.SOURCE)
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        train = apply_durable_change_train(conn, train)
    assert train.apply_evidence is not None

    late_post = replace(
        train.apply_evidence.post,
        observed_at_ms=train.apply_evidence.applied_at_ms + 1,
    )
    invalid_apply = replace(
        train,
        apply_evidence=replace(train.apply_evidence, post=late_post),
    )
    with pytest.raises(DurableChangeTrainError, match="apply timestamp predates post-apply"):
        migration_runner.validate_durable_change_train_manifest(invalid_apply)

    release_time = train.apply_evidence.applied_at_ms + 100
    released = record_durable_writer_release(
        train,
        evidence_ref="proof:lease-release",
        released_at_ms=release_time,
    )
    assert released.reservation is not None
    invalid_release = replace(
        released,
        reservation=replace(
            released.reservation,
            released_at_ms=train.apply_evidence.applied_at_ms - 1,
        ),
    )
    with pytest.raises(DurableChangeTrainError, match="writer release timestamp predates"):
        migration_runner.validate_durable_change_train_manifest(invalid_release)

    with sqlite3.connect(db_path) as restarted:
        with _memory_target() as fresh:
            parity = prove_durable_fresh_ddl_parity(
                ArchiveTier.SOURCE,
                _TARGET_VERSION,
                migrated_connection=restarted,
                fresh_connection=fresh,
                evidence_ref="proof:actual-fresh",
            )
        runtime_results = _runtime_results()
        restart = capture_durable_restart_convergence(
            restarted,
            released,
            runtime_consumers=runtime_results,
            evidence_ref="proof:restart",
        )
    restart_before_release = replace(
        restart,
        observed_at_ms=train.apply_evidence.applied_at_ms + 50,
    )
    with pytest.raises(DurableChangeTrainError, match="restart convergence timestamp predates writer release"):
        prove_durable_change_train(
            released,
            fresh_ddl_parity=parity,
            runtime_consumers=runtime_results,
            restart_convergence=restart_before_release,
            proven_at_ms=release_time + 1,
        )


def test_manifest_checksum_revision_and_unsafe_path_are_enforced(tmp_path: Path) -> None:
    train = _declared(ArchiveTier.SOURCE)
    path = tmp_path / "train.json"
    write_durable_change_train_manifest(path, train, expected_revision=-1)
    with pytest.raises(DurableChangeTrainError, match="revision changed"):
        write_durable_change_train_manifest(path, train, expected_revision=99)
    with pytest.raises(DurableChangeTrainError, match="advance exactly one revision"):
        write_durable_change_train_manifest(path, train, expected_revision=0)
    skipped_revision = replace(train, revision=2)
    with pytest.raises(DurableChangeTrainError, match="advance exactly one revision"):
        write_durable_change_train_manifest(path, skipped_revision, expected_revision=0)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["owner_ref"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DurableChangeTrainError, match="checksum mismatch"):
        load_durable_change_train_manifest(path)

    real = tmp_path / "real.json"
    write_durable_change_train_manifest(real, train, expected_revision=-1)
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(DurableChangeTrainError, match="not a real single-linked file"):
        load_durable_change_train_manifest(link)

    parent = tmp_path / "real-parent"
    parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(parent, target_is_directory=True)
    with pytest.raises(DurableChangeTrainError, match="parent.*symbolic link"):
        write_durable_change_train_manifest(linked_parent / "train.json", train, expected_revision=-1)

    locked_path = tmp_path / "locked-train.json"
    lock_path = tmp_path / ".locked-train.json.lock"
    lock_target = tmp_path / "lock-target"
    lock_target.write_text("not a lock", encoding="utf-8")
    lock_path.symlink_to(lock_target)
    with pytest.raises(DurableChangeTrainError, match="manifest lock safely"):
        write_durable_change_train_manifest(locked_path, train, expected_revision=-1)


def test_rechecks_manifest_semantics_after_a_valid_checksum(tmp_path: Path) -> None:
    train = _admitted(ArchiveTier.USER)
    payload = migration_runner.durable_change_train_to_payload(train)
    parity = payload["fresh_ddl_parity"]
    assert isinstance(parity, dict)
    parity["matches"] = False
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256")
    payload["manifest_sha256"] = migration_runner._canonical_json_sha256(unsigned)
    path = tmp_path / "forged-train.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DurableChangeTrainError, match="fresh-DDL parity is not an exact match"):
        load_durable_change_train_manifest(path)


def _adopt_audit_into_a_real_archive(archive_root: Path, *, owner_id: str) -> Path:
    """Adopt a missing audit tier through the real operator route."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(archive_root)
    (archive_root / "audit.db").unlink()
    backup = backup_archive(
        output_dir=archive_root.parent / f"backup-{owner_id.replace(':', '-')}",
        profile="full_evidence",
        verify=True,
    )
    assert backup.ok, backup.error
    assert backup.output_path is not None
    with acquire_durable_archive_ownership(archive_root, owner_id=owner_id) as owner:
        _version, receipt = adopt_missing_audit_tier(
            archive_root / "audit.db",
            backup_manifest=Path(backup.output_path) / "manifest.json",
            directory_fd=owner.directory_fd,
            stopped_daemon_check=lambda: "proof:test-daemon-stopped",
        )
    return receipt


def _rewrite_durable_tier_file(path: Path) -> None:
    """Republish one durable tier the way a durable migration does.

    ``os.replace`` of a rebuilt file is what changes the tier's inode, which is
    the archive-identity change that stranded the live adoption receipt.
    """
    rebuilt = path.with_name(f"{path.name}.rebuilt")
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("VACUUM INTO ?", (str(rebuilt),))
    before = path.stat().st_ino
    os.replace(rebuilt, path)
    assert path.stat().st_ino != before


def test_audit_adoption_is_resealed_across_a_durable_tier_rewrite(workspace_env: dict[str, Path]) -> None:
    """A released durable rewrite of source.db does not strand the adoption seal.

    Anti-vacuity: dropping the rebind chain from ``_load_audit_adoption_receipt``
    makes this fail with ``audit adoption receipt source/user authority
    mismatch``, which is the live archive's observed refusal.
    """
    from polylogue.operations.durable_change_train import (
        _rebind_audit_adoption_after_durable_rewrite,
        audit_adoption_sealed_authority_digest,
    )
    from polylogue.storage.sqlite.durable_change_train import _durable_chain_floor_versions

    archive_root = workspace_env["archive_root"]
    _adopt_audit_into_a_real_archive(archive_root, owner_id="test:audit-reseal")
    assert validate_audit_adoption_receipt(archive_root) == audit_adoption_receipt_path(archive_root)

    sealed = audit_adoption_sealed_authority_digest(archive_root)
    assert sealed is not None
    _rewrite_durable_tier_file(archive_root / "source.db")

    with pytest.raises(MigrationError, match="audit adoption receipt source/user authority mismatch"):
        validate_audit_adoption_receipt(archive_root)

    record = _rebind_audit_adoption_after_durable_rewrite(
        archive_root,
        sealed_digest=sealed,
        proof_ref="proof:durable-change-train:source",
    )
    assert record is not None and record.is_file()
    assert validate_audit_adoption_receipt(archive_root) == audit_adoption_receipt_path(archive_root)
    # Relocation reads the adopted chain floor only after that gate passes.
    # It refuses fresh-bootstrap train authority outright, so the marker is
    # absent on every archive that reaches the floor read (as in the relocation
    # suite's own adopted-archive fixtures).
    (archive_root / ".maintenance-state" / "durable-change-trains" / ".bootstrap").unlink()
    assert (
        _durable_chain_floor_versions(archive_root, archive_root / ".maintenance-state" / "durable-change-trains")[
            ArchiveTier.AUDIT
        ]
        == ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT]
    )
    # The chain is idempotent: a second rebind at an unchanged identity is a no-op.
    resealed = audit_adoption_sealed_authority_digest(archive_root)
    assert resealed is not None
    assert (
        _rebind_audit_adoption_after_durable_rewrite(
            archive_root,
            sealed_digest=resealed,
            proof_ref="proof:durable-change-train:source",
        )
        is None
    )


def test_durable_change_train_execution_carries_the_adoption_seal(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production migration route re-seals adoption across its own rewrite.

    Anti-vacuity: removing the re-seal from ``execute_durable_change_train``
    leaves the receipt stranded and this raises the authority mismatch.
    """
    import polylogue.operations.durable_change_train as operations_durable_change_train

    archive_root = workspace_env["archive_root"]
    _adopt_audit_into_a_real_archive(archive_root, owner_id="test:audit-train-seal")

    def rewrite_instead_of_migrating(root: Path, _tier: ArchiveTier, **_kwargs: object) -> object:
        _rewrite_durable_tier_file(root / "source.db")
        return SimpleNamespace(train=None, migration_result=None)

    monkeypatch.setattr(
        operations_durable_change_train,
        "_execute_durable_change_train",
        rewrite_instead_of_migrating,
    )
    operations_durable_change_train.execute_durable_change_train(
        archive_root,
        ArchiveTier.SOURCE,
        backup_manifest=None,
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
        release_archive_ownership=lambda: None,
    )

    assert validate_audit_adoption_receipt(archive_root) == audit_adoption_receipt_path(archive_root)


def test_audit_adoption_refuses_a_receipt_from_a_different_archive(
    workspace_env: dict[str, Path], tmp_path: Path
) -> None:
    """A foreign adopted audit image and its re-sealed chain are still refused.

    This is the protection the rebind chain must not erase: transplanting the
    whole adoption ledger names the *donor* archive's durable identity, and no
    automatic route on the recipient can continue that seal.
    """
    from polylogue.operations.durable_change_train import (
        _rebind_audit_adoption_after_durable_rewrite,
        audit_adoption_sealed_authority_digest,
    )

    donor_root = workspace_env["archive_root"]
    _adopt_audit_into_a_real_archive(donor_root, owner_id="test:audit-donor")
    sealed = audit_adoption_sealed_authority_digest(donor_root)
    assert sealed is not None
    _rewrite_durable_tier_file(donor_root / "source.db")
    assert (
        _rebind_audit_adoption_after_durable_rewrite(
            donor_root, sealed_digest=sealed, proof_ref="proof:durable-change-train:source"
        )
        is not None
    )

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    recipient_root = tmp_path / "recipient"
    recipient_root.mkdir()
    initialize_active_archive_root(recipient_root)
    donor_ledger = donor_root / ".maintenance-state" / "durable-change-trains"
    recipient_ledger = recipient_root / ".maintenance-state" / "durable-change-trains"
    shutil.copy2(donor_root / "audit.db", recipient_root / "audit.db")
    for name in os.listdir(donor_ledger):
        if name.startswith("audit-"):
            shutil.copy2(donor_ledger / name, recipient_ledger / name)

    with pytest.raises(MigrationError, match="audit adoption receipt source/user authority mismatch"):
        validate_audit_adoption_receipt(recipient_root)
    # The self-proving route cannot launder it either: the recipient never held
    # the donor's seal, so it can never observe it before its own rewrite.
    assert audit_adoption_sealed_authority_digest(recipient_root) is None
    with pytest.raises(MigrationError, match="does not continue the sealed source/user authority"):
        _rebind_audit_adoption_after_durable_rewrite(
            recipient_root, sealed_digest=sealed, proof_ref="proof:durable-change-train:source"
        )


def test_durable_change_train_execution_carries_the_bootstrap_seal(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durable rewrite does not strand a fresh archive's own bootstrap marker.

    The marker seals the same inode identity the adoption receipt does, so the
    first released migration of a directly bootstrapped archive would otherwise
    make every later startup reconcile refuse it.

    Anti-vacuity: removing the re-seal from ``execute_durable_change_train``
    raises ``fresh durable bootstrap marker durable identity mismatch`` from
    ``reconcile_durable_change_train_startup``.
    """
    import polylogue.operations.durable_change_train as operations_durable_change_train
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    marker = archive_root / ".maintenance-state" / "durable-change-trains" / ".bootstrap"
    assert marker.is_file()
    sealed_versions = json.loads(marker.read_text(encoding="utf-8"))["versions"]

    def rewrite_instead_of_migrating(root: Path, _tier: ArchiveTier, **_kwargs: object) -> object:
        _rewrite_durable_tier_file(root / "source.db")
        return SimpleNamespace(train=None, migration_result=None)

    monkeypatch.setattr(
        operations_durable_change_train,
        "_execute_durable_change_train",
        rewrite_instead_of_migrating,
    )
    operations_durable_change_train.execute_durable_change_train(
        archive_root,
        ArchiveTier.SOURCE,
        backup_manifest=None,
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
        release_archive_ownership=lambda: None,
    )

    assert reconcile_durable_change_train_startup(archive_root) == ()
    # The marker's authority is the recorded bootstrap versions, not its seal.
    assert json.loads(marker.read_text(encoding="utf-8"))["versions"] == sealed_versions


def test_a_manifest_written_before_the_projected_digest_still_verifies() -> None:
    """A recorded train stays verifiable across the parity-digest split.

    polylogue-jkoah separated ``DurableFreshDDLParityProof``'s projected digest
    from its unprojected one. Every manifest written before that split carries
    a single digest, and it is necessarily both: the split only separates two
    values on a tier that retains a declared retirement, and such a tier could
    not have been recorded while the proof that consumes the parity refused it.
    So the decoder completes the missing slot from the recorded digest rather
    than declaring the manifest unverifiable.

    The manifest checksum is verified before the backfill, so this cannot be
    used to admit an unauthenticated payload.

    Anti-vacuity: without the decoder backfill this raises
    ``train.fresh_ddl_parity fields differ: missing=['parity_inventory_sha256']``.
    A manifest carrying the field is decoded untouched, which the second half
    pins.
    """
    train = _admitted(ArchiveTier.SOURCE)
    payload = migration_runner.durable_change_train_to_payload(train)
    parity_payload = payload["fresh_ddl_parity"]
    assert isinstance(parity_payload, dict)
    assert "parity_inventory_sha256" in parity_payload

    # Exactly the bytes a pre-split writer produced: the field absent, and the
    # checksum computed over the payload without it.
    legacy = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    legacy["fresh_ddl_parity"] = {
        key: value for key, value in parity_payload.items() if key != "parity_inventory_sha256"
    }
    legacy["manifest_sha256"] = migration_runner._canonical_json_sha256(legacy)

    decoded = migration_runner.durable_change_train_from_payload(legacy)
    assert decoded.fresh_ddl_parity is not None
    assert decoded.fresh_ddl_parity.parity_inventory_sha256 == decoded.fresh_ddl_parity.migrated_inventory_sha256
    assert decoded == train

    # A manifest that already carries the field is not rewritten by the backfill.
    assert migration_runner.durable_change_train_from_payload(payload) == train

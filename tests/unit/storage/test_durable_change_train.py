"""Lifecycle contracts for durable source/user schema-change authority."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from polylogue.storage.sqlite import migration_runner
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import (
    DURABLE_MIGRATION_ADOPTION_FLOORS,
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
    MigrationStep,
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

_CURRENT_VERSION = 1
_TARGET_VERSION = 2
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
        fresh_ddl_parity=_parity(tier),
        admission_evidence_ref=f"proof:admit:{tier.value}",
        active_trains=active_trains,
        migration_claims=(migration,),
        canonical_target_version=_TARGET_VERSION,
        admitted_at_ms=2,
    )


def _install_synthetic_migration(
    monkeypatch: pytest.MonkeyPatch,
    tier: ArchiveTier,
    *,
    sql: str = _ADDITIVE_SQL,
) -> None:
    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[tier] = _TARGET_VERSION
    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)
    step = MigrationStep(
        tier=tier,
        version=_TARGET_VERSION,
        name="002_durable_items.sql",
        sql=sql,
        requires_backup=not sql.startswith("-- migration-safety: additive-no-backup"),
    )
    monkeypatch.setattr(
        migration_runner,
        "_load_migrations",
        lambda observed_tier: (step,) if observed_tier is tier else (),
    )


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
    _install_synthetic_migration(monkeypatch, tier)
    original_runner = migration_runner.migrate_archive_tier
    calls: list[ArchiveTier] = []

    def observed_runner(
        conn: sqlite3.Connection,
        observed_tier: ArchiveTier,
        *,
        backup_manifest: Path | None,
    ) -> migration_runner.MigrationResult:
        calls.append(observed_tier)
        return original_runner(conn, observed_tier, backup_manifest=backup_manifest)

    def persist_and_reload(next_train: DurableChangeTrain, expected_revision: int) -> DurableChangeTrain:
        write_durable_change_train_manifest(
            manifest,
            next_train,
            expected_revision=expected_revision,
        )
        return load_durable_change_train_manifest(manifest)

    monkeypatch.setattr(migration_runner, "migrate_archive_tier", observed_runner)
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
    assert calls == [tier]

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
    sql_path = source_package / "027_future_items.sql"
    sql_path.write_text(sql, encoding="utf-8")
    claim = durable_migration_claim_for_sql(
        ArchiveTier.SOURCE,
        sql_path.name,
        sql,
        owner_ref="owner:future-source",
    )
    train = declare_durable_change_train(
        train_id="train:source:v27",
        tier=ArchiveTier.SOURCE,
        current_version=DURABLE_MIGRATION_ADOPTION_FLOORS[ArchiveTier.SOURCE],
        target_version=27,
        slot=27,
        owner_ref="owner:future-source",
        migration=claim,
        riders=(_rider(),),
        declared_at_ms=1,
    )
    (source_package / "027.train.json").write_text(
        json.dumps(migration_runner.durable_change_train_to_payload(train)),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda _tier: "fixture_migrations.source",
    )

    observed = validate_durable_migration_sidecars(ArchiveTier.SOURCE, ((sql_path.name, sql),))

    assert [item.resource_name for item in observed] == ["027.train.json"]
    assert observed[0].train.migration.sql_sha256 == claim.sql_sha256
    assert "fixture_migrations.source" in sys.modules


def test_future_train_sidecar_hash_and_slot_are_admission_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "fixture_migrations_hash"
    source_package = package_root / "source"
    source_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (source_package / "__init__.py").write_text("", encoding="utf-8")
    sql = "-- migration-safety: additive-no-backup\nCREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;\n"
    sql_path = source_package / "027_future_items.sql"
    sql_path.write_text(sql, encoding="utf-8")
    claim = durable_migration_claim_for_sql(ArchiveTier.SOURCE, sql_path.name, sql, owner_ref="owner:future-source")
    train = declare_durable_change_train(
        train_id="train:source:v27",
        tier=ArchiveTier.SOURCE,
        current_version=26,
        target_version=27,
        slot=27,
        owner_ref="owner:future-source",
        migration=claim,
        riders=(_rider(),),
        declared_at_ms=1,
    )
    payload = migration_runner.durable_change_train_to_payload(train)
    cast(dict[str, object], payload["migration"])["sql_sha256"] = "0" * 64
    (source_package / "027.train.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda _tier: "fixture_migrations_hash.source",
    )

    with pytest.raises(DurableChangeTrainError, match="checksum mismatch"):
        validate_durable_migration_sidecars(ArchiveTier.SOURCE, ((sql_path.name, sql),))


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


def test_source_008_009_collision_names_both_owners_and_blocks_admission() -> None:
    source_migrations = Path(__file__).parents[3] / "polylogue" / "storage" / "sqlite" / "migrations" / "source"
    source_008 = source_migrations / "008_raw_session_capture_mode.sql"
    source_009 = source_migrations / "009_expand_origin_vocabulary.sql"
    first = durable_migration_claim_for_sql(
        ArchiveTier.SOURCE,
        source_008.name,
        source_008.read_text(encoding="utf-8"),
        owner_ref="owner:source-008",
    )
    late_rider = durable_migration_claim_for_sql(
        ArchiveTier.SOURCE,
        "008_expand_origin_vocabulary.sql",
        source_009.read_text(encoding="utf-8"),
        owner_ref="owner:source-009-late-rider",
    )
    report = durable_migration_collision_report((first, late_rider))
    assert report["ok"] is False
    serialized = json.dumps(report)
    assert source_008.name in serialized
    assert "008_expand_origin_vocabulary.sql" in serialized
    assert "owner:source-008" in serialized
    assert "owner:source-009-late-rider" in serialized

    train = declare_durable_change_train(
        train_id="source-v8",
        tier=ArchiveTier.SOURCE,
        current_version=7,
        target_version=8,
        slot=8,
        owner_ref="owner:source-train",
        migration=first,
        riders=(_rider(),),
    )
    parity = DurableFreshDDLParityProof(
        tier=ArchiveTier.SOURCE,
        target_version=8,
        migrated_version=8,
        fresh_version=8,
        migrated_inventory_sha256="a" * 64,
        fresh_inventory_sha256="a" * 64,
        missing_objects=(),
        unexpected_objects=(),
        changed_objects=(),
        evidence_ref="proof:v8-fresh",
        matches=True,
    )
    with pytest.raises(DurableChangeTrainError, match="collision.*rebase/renumber") as exc_info:
        admit_durable_change_train(
            train,
            observed_current_version=7,
            fresh_ddl_parity=parity,
            admission_evidence_ref="proof:v8-admit",
            migration_claims=(first, late_rider),
            canonical_target_version=8,
        )
    assert source_008.name in str(exc_info.value)
    assert "008_expand_origin_vocabulary.sql" in str(exc_info.value)


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
    failing_sql = """-- migration-safety: additive-no-backup
CREATE TABLE durable_items (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL) STRICT;
INSERT INTO table_that_does_not_exist VALUES (1);
"""
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    claim = _claim(ArchiveTier.SOURCE, failing_sql)
    train = _admitted(ArchiveTier.SOURCE, claim=claim)
    _install_synthetic_migration(monkeypatch, ArchiveTier.SOURCE, sql=failing_sql)
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
    _install_synthetic_migration(monkeypatch, ArchiveTier.USER)
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
    _install_synthetic_migration(monkeypatch, ArchiveTier.SOURCE)
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

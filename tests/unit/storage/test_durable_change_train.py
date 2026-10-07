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
from typing import Any, cast

import pytest

import polylogue.storage.sqlite.durable_change_train as durable_change_train_module
from polylogue.storage.sqlite import migration_runner
from polylogue.storage.sqlite.archive_tiers import (
    ARCHIVE_BASELINE_DDL_BY_TIER,
    ARCHIVE_DDL_BY_TIER,
    ARCHIVE_VERSION_BY_TIER,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import (
    DURABLE_MIGRATION_ADOPTION_FLOORS,
    _runtime_consumer_results,
    durable_change_train_manifest_path,
    durable_change_train_policy_report,
    durable_train_manifest_paths,
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
    DurableMigrationClaim,
    DurableMigrationReplayProof,
    DurableMigrationReplayStep,
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
    reconcile_interrupted_durable_change_train,
    record_durable_writer_release,
    recover_durable_change_train,
    rehearse_durable_migration_chain,
    release_durable_change_train,
    reserve_durable_change_train,
    write_durable_change_train_manifest,
)
from tests.infra.durable_schema_reset import reset_source_fixture_to_version
from tests.infra.durable_tier_fixtures import refresh_archive_format_marker

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
_DATA_DEPENDENT_FAILURE_SQL = """-- migration-safety: additive-no-backup
CREATE UNIQUE INDEX base_items_payload_unique ON base_items(payload);
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


@contextmanager
def _pinned_runtime_target(tier: ArchiveTier, target: int) -> Iterator[None]:
    """Give isolated lifecycle fixtures the package target they declare."""
    previous = migration_runner.ARCHIVE_VERSION_BY_TIER
    versions = dict(previous)
    versions[tier] = target
    migration_runner.ARCHIVE_VERSION_BY_TIER = versions
    try:
        yield
    finally:
        migration_runner.ARCHIVE_VERSION_BY_TIER = previous


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


def _parity(
    tier: ArchiveTier,
    *,
    include_durable_items: bool = True,
    matches: bool | None = None,
    claim: DurableMigrationClaim | None = None,
) -> DurableMigrationReplayProof:
    canonical_target = max(migration_runner.ARCHIVE_VERSION_BY_TIER[tier], _TARGET_VERSION)
    if include_durable_items and canonical_target == migration_runner.ARCHIVE_VERSION_BY_TIER[tier]:
        with sqlite3.connect(":memory:") as source:
            source.executescript(_BASE_ITEMS_DDL)
            source.execute(f"PRAGMA user_version = {_CURRENT_VERSION}")
            source.commit()
            try:
                replay = rehearse_durable_migration_chain(
                    source,
                    tier,
                    target_version=migration_runner.ARCHIVE_VERSION_BY_TIER[tier],
                    evidence_ref=f"proof:schema-replay:{tier.value}",
                )
                if replay.matches:
                    return replay
            except (MigrationError, DurableChangeTrainError, sqlite3.Error):
                pass
    claim = claim or _claim(tier)
    before = hashlib.sha256(f"fixture:{tier.value}:v{_CURRENT_VERSION}".encode()).hexdigest()
    steps: list[DurableMigrationReplayStep] = []
    for version in range(_CURRENT_VERSION + 1, canonical_target + 1):
        after = hashlib.sha256(f"fixture:{tier.value}:v{version}".encode()).hexdigest()
        steps.append(
            DurableMigrationReplayStep(
                version=version,
                name=Path(claim.path).name if version == _TARGET_VERSION else f"{version:03d}_fixture.sql",
                sql_sha256=(
                    claim.sql_sha256
                    if version == _TARGET_VERSION
                    else hashlib.sha256(f"fixture-sql:{tier.value}:{version}".encode()).hexdigest()
                ),
                before_schema_inventory_sha256=before,
                after_schema_inventory_sha256=after,
            )
        )
        before = after
    proof_matches = include_durable_items if matches is None else matches
    terminal = steps[-1].after_schema_inventory_sha256
    canonical = terminal if proof_matches else hashlib.sha256(b"different-canonical-fixture").hexdigest()
    proof = DurableMigrationReplayProof(
        tier=tier,
        from_version=_CURRENT_VERSION,
        target_version=canonical_target,
        original_schema_inventory_sha256=steps[0].before_schema_inventory_sha256,
        steps=tuple(steps),
        terminal_schema_inventory_sha256=terminal,
        canonical_version=canonical_target,
        canonical_schema_inventory_sha256=canonical,
        chain_sha256="",
        evidence_ref=f"proof:schema-replay:{tier.value}",
        matches=proof_matches,
    )
    return replace(
        proof,
        chain_sha256=migration_runner._migration_replay_chain_digest(
            tier=proof.tier,
            from_version=proof.from_version,
            target_version=proof.target_version,
            original_schema_inventory_sha256=proof.original_schema_inventory_sha256,
            steps=proof.steps,
            terminal_schema_inventory_sha256=proof.terminal_schema_inventory_sha256,
            canonical_version=proof.canonical_version,
            canonical_schema_inventory_sha256=proof.canonical_schema_inventory_sha256,
        ),
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
    parity: DurableMigrationReplayProof | None = None,
) -> DurableChangeTrain:
    migration = claim or _claim(tier)
    replay = parity if parity is not None else _parity(tier, claim=migration)
    with _pinned_runtime_target(tier, replay.target_version):
        return _admit_for_test(
            _declared(
                tier,
                claim=migration,
                rider=rider,
                owner_ref=owner_ref,
                backup_plan_ref=backup_plan_ref,
            ),
            observed_current_version=_CURRENT_VERSION,
            schema_replay_proof=replay,
            admission_evidence_ref=f"proof:admit:{tier.value}",
            active_trains=active_trains,
            migration_claims=(migration,),
            canonical_target_version=_TARGET_VERSION,
            admitted_at_ms=2,
        )


def _admit_for_test(train: DurableChangeTrain, **kwargs: object) -> DurableChangeTrain:
    replay = kwargs.get("schema_replay_proof")
    if not isinstance(replay, DurableMigrationReplayProof):
        raise AssertionError("synthetic train admission requires its replay proof")
    with _pinned_runtime_target(train.tier, replay.target_version):
        return admit_durable_change_train(train, **kwargs)  # type: ignore[arg-type]


def _unrelated_replay(proof: DurableMigrationReplayProof) -> DurableMigrationReplayProof:
    shifted = replace(proof.steps[-1], version=4)
    unrelated = replace(
        proof,
        from_version=3,
        target_version=4,
        steps=(shifted,),
        canonical_version=4,
        chain_sha256="",
    )
    return replace(
        unrelated,
        chain_sha256=migration_runner._migration_replay_chain_digest(
            tier=unrelated.tier,
            from_version=unrelated.from_version,
            target_version=unrelated.target_version,
            original_schema_inventory_sha256=unrelated.original_schema_inventory_sha256,
            steps=unrelated.steps,
            terminal_schema_inventory_sha256=unrelated.terminal_schema_inventory_sha256,
            canonical_version=unrelated.canonical_version,
            canonical_schema_inventory_sha256=unrelated.canonical_schema_inventory_sha256,
        ),
    )


#: The synthetic fixtures below own slot ``002`` -- the first slot any durable
#: tier of this lineage may own, because ``ARCHIVE_FORMAT_FLOOR_VERSION`` is 1.
_SYNTHETIC_SIDECAR_NAME = f"{_TARGET_VERSION:03d}.train.json"

_SOURCE_ADOPTION_FLOOR = DURABLE_MIGRATION_ADOPTION_FLOORS[ArchiveTier.SOURCE]
# The first slot a source train may own. Synthetic future-migration fixtures
# must sit above the floor, or sidecar discovery refuses them before the
# behavior under test runs.
_NEXT_SOURCE_SLOT = _SOURCE_ADOPTION_FLOOR + 1
#: Fresh v1 (#5551) ships every durable tier at its adoption floor, so a
#: bootstrap marker grants nothing. The tests below need a real, shipped
#: above-floor schema on the source and user tiers they exercise, so they run
#: only once every durable tier ships a numbered migration; one tier moving
#: alone does not make their premise true.
_SOME_DURABLE_TIER_AT_FLOOR = any(
    ARCHIVE_VERSION_BY_TIER[tier] <= DURABLE_MIGRATION_ADOPTION_FLOORS[tier]
    for tier in DURABLE_MIGRATION_ADOPTION_FLOORS
)
_needs_shipped_durable_slot = pytest.mark.skipif(
    _SOME_DURABLE_TIER_AT_FLOOR,
    reason="a durable tier ships no numbered migration above its adoption floor, so the premise is unconstructible",
)
_NEXT_SOURCE_SQL_NAME = f"{_NEXT_SOURCE_SLOT:03d}_future_items.sql"
_NEXT_SOURCE_SIDECAR_NAME = f"{_NEXT_SOURCE_SLOT:03d}.train.json"


#: Every module that rebinds the durable version map at import time, plus the
#: package attribute the lazily-importing durable-train helpers read. A test
#: that pins only one of them leaves the others disagreeing about the tier's
#: target, which no production configuration ever does.
_VERSION_MAP_OWNERS = (
    "polylogue.storage.sqlite.archive_tiers.ARCHIVE_VERSION_BY_TIER",
    "polylogue.storage.sqlite.archive_tiers.bootstrap.ARCHIVE_VERSION_BY_TIER",
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
        riders=(_production_rider(),),
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
    from polylogue.storage.sqlite import archive_tiers

    baseline = dict(ARCHIVE_BASELINE_DDL_BY_TIER)
    baseline[tier] = canonical_base
    monkeypatch.setattr(archive_tiers, "ARCHIVE_BASELINE_DDL_BY_TIER", baseline)
    canonical = f"{canonical_base}\n{sql}"
    ddl[tier] = canonical
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)
    monkeypatch.setattr(bootstrap, "ARCHIVE_DDL_BY_TIER", ddl)
    canonical_package = migration_runner._migration_package
    monkeypatch.setattr(
        migration_runner,
        "_migration_package",
        lambda observed: f"{package_name}.{tier.value}" if observed is tier else canonical_package(observed),
    )
    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train._migration_package",
        lambda observed: f"{package_name}.{tier.value}" if observed is tier else canonical_package(observed),
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
    _install_synthetic_migration(
        tmp_path,
        monkeypatch,
        ArchiveTier.SOURCE,
        canonical_base=ARCHIVE_BASELINE_DDL_BY_TIER[ArchiveTier.SOURCE],
    )
    with sqlite3.connect(db_path) as live:
        replay = rehearse_durable_migration_chain(
            live,
            ArchiveTier.SOURCE,
            target_version=_TARGET_VERSION,
            evidence_ref="proof:source-hook-event-replay",
        )
    train = _admitted(ArchiveTier.SOURCE, rider=_source_hook_event_production_rider(), parity=replay)
    with sqlite3.connect(db_path) as conn:
        train = _reserve_and_authorize(conn, train, archive_root=tmp_path)
        train = apply_durable_change_train(conn, train)

    train = record_durable_writer_release(train, evidence_ref="proof:source-hook-event-writer-release")
    with sqlite3.connect(db_path) as restarted:
        actual_parity = train.schema_replay_proof
        assert actual_parity is not None
        runtime_results = _runtime_consumer_results(train, tmp_path, candidate=restarted)
        restart = capture_durable_restart_convergence(
            restarted,
            train,
            runtime_consumers=runtime_results,
            evidence_ref="proof:source-hook-event-restart",
        )
    train = prove_durable_change_train(
        train,
        schema_replay_proof=actual_parity,
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
        schema_replay_proof=_parity(tier),
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
        actual_parity = train.schema_replay_proof
        assert actual_parity is not None
        runtime_results = _runtime_results()
        restart = capture_durable_restart_convergence(
            restarted,
            train,
            runtime_consumers=runtime_results,
            evidence_ref="proof:runtime-restart",
        )
    unrelated = _unrelated_replay(actual_parity)
    with pytest.raises(DurableChangeTrainError, match="schema replay does not bind numbered migration"):
        prove_durable_change_train(
            train,
            schema_replay_proof=unrelated,
            runtime_consumers=runtime_results,
            restart_convergence=restart,
        )
    previous_revision = train.revision
    train = prove_durable_change_train(
        train,
        schema_replay_proof=actual_parity,
        runtime_consumers=runtime_results,
        restart_convergence=restart,
    )
    train = persist_and_reload(train, previous_revision)
    previous_revision = train.revision
    train = release_durable_change_train(train, evidence_ref="proof:train-release")
    train = persist_and_reload(train, previous_revision)

    assert train.proof is not None
    invalid_proof = replace(train.proof, schema_replay_proof=unrelated)
    invalid_train = replace(train, proof=invalid_proof)
    with pytest.raises(DurableChangeTrainError, match="schema replay does not bind numbered migration"):
        write_durable_change_train_manifest(manifest, invalid_train, expected_revision=train.revision)

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
        riders=(_production_rider(),),
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
    from polylogue.storage.sqlite import archive_tiers

    baseline = dict(ARCHIVE_BASELINE_DDL_BY_TIER)
    baseline[ArchiveTier.SOURCE] = _BASE_ITEMS_DDL
    monkeypatch.setattr(archive_tiers, "ARCHIVE_BASELINE_DDL_BY_TIER", baseline)
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
    ddl[ArchiveTier.SOURCE] = ARCHIVE_BASELINE_DDL_BY_TIER[ArchiveTier.SOURCE] + "\n" + sql
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


def test_maintenance_route_rehearses_an_intermediate_sidecar_to_the_shipped_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A v1 archive rehearses through final v3 DDL, then commits one slot at a time."""
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
        (4, "final_items", "CREATE TABLE final_items (id INTEGER PRIMARY KEY) STRICT;"),
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
    # Ship both steps in the fixture, but first run the v2 runtime against its
    # own canonical DDL.  The following phase upgrades the runtime to v3 while
    # the persisted v2 train remains released, exercising startup's historical
    # proof path rather than a helper-only validator.
    _pin_source_runtime_version(monkeypatch, 2)
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    ddl = dict(ARCHIVE_DDL_BY_TIER)
    ddl[ArchiveTier.SOURCE] = "\n".join((ARCHIVE_BASELINE_DDL_BY_TIER[ArchiveTier.SOURCE], migrations[0][2]))
    monkeypatch.setattr(bootstrap, "ARCHIVE_DDL_BY_TIER", ddl)
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", ddl)
    db_path = tmp_path / "source.db"

    first = execute_durable_change_train(
        tmp_path,
        ArchiveTier.SOURCE,
        backup_manifest=None,
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
        release_archive_ownership=lambda: None,
    )
    released_v2_path = durable_change_train_manifest_path(tmp_path, ArchiveTier.SOURCE, 2)
    released_v2 = load_durable_change_train_manifest(released_v2_path)
    assert released_v2.state is DurableChangeTrainState.RELEASED

    _pin_source_runtime_version(monkeypatch, 4)
    final_ddl = dict(ddl)
    final_ddl[ArchiveTier.SOURCE] = "\n".join((ddl[ArchiveTier.SOURCE], migrations[1][2], migrations[2][2]))
    monkeypatch.setattr(bootstrap, "ARCHIVE_DDL_BY_TIER", final_ddl)
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", final_ddl)

    # This is the production startup consumer.  Reinstating the old rule that
    # equated a historical witness's terminal identity with today's canonical
    # DDL rejects this persisted v2 train before the v3 step can be admitted.
    assert reconcile_durable_change_train_startup(tmp_path) == (released_v2_path,)
    assert load_durable_change_train_manifest(released_v2_path).state is DurableChangeTrainState.RELEASED

    step_2_path = source_package / "002_durable_items.sql"
    original_step_2 = step_2_path.read_text(encoding="utf-8")
    step_2_path.write_text(
        "-- migration-safety: additive-no-backup\nCREATE TABLE altered_items (id INTEGER PRIMARY KEY) STRICT;\n",
        encoding="utf-8",
    )
    try:
        with pytest.raises(DurableChangeTrainError, match="sidecar SQL SHA-256 mismatch"):
            reconcile_durable_change_train_startup(tmp_path)
    finally:
        step_2_path.write_text(original_step_2, encoding="utf-8")

    second = execute_durable_change_train(
        tmp_path,
        ArchiveTier.SOURCE,
        backup_manifest=None,
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
        release_archive_ownership=lambda: None,
    )
    released_v3_path = durable_change_train_manifest_path(tmp_path, ArchiveTier.SOURCE, 3)
    assert reconcile_durable_change_train_startup(tmp_path) == (released_v2_path, released_v3_path)

    # The v3 train's witnessed intermediate inventory remains the authority
    # while today's runtime DDL already describes v4.  An undeclared object
    # at v3 must therefore refuse startup before v4 is applied.
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE unexpected_intermediate (id INTEGER PRIMARY KEY) STRICT")
        conn.commit()
    with pytest.raises(DurableChangeTrainError, match="differs from the released migration replay witness"):
        reconcile_durable_change_train_startup(tmp_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE unexpected_intermediate")
        conn.commit()

    third = execute_durable_change_train(
        tmp_path,
        ArchiveTier.SOURCE,
        backup_manifest=None,
        daemon_stopped_evidence_ref="proof:daemon-stopped",
        single_writer_evidence_ref="proof:archive-ownership-lock",
        release_archive_ownership=lambda: None,
    )
    assert first.train is not None and first.train.target_version == 2
    assert second.train is not None and second.train.target_version == 3
    assert third.train is not None and third.train.target_version == 4
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone() == (4,)
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='durable_items'").fetchone() == (
            "durable_items",
        )
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='later_items'").fetchone() == ("later_items",)
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='final_items'").fetchone() == ("final_items",)


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
        SimpleNamespace(
            state=DurableChangeTrainState.RELEASED,
            tier=ArchiveTier.SOURCE,
            target_version=first_slot,
            schema_replay_proof=None,
        ),
    )
    later_released = cast(
        DurableChangeTrain,
        SimpleNamespace(
            state=DurableChangeTrainState.RELEASED,
            tier=ArchiveTier.SOURCE,
            target_version=later_slot,
            schema_replay_proof=None,
        ),
    )
    backup_authorized = cast(
        DurableChangeTrain,
        SimpleNamespace(
            state=DurableChangeTrainState.BACKUP_AUTHORIZED,
            tier=ArchiveTier.SOURCE,
            target_version=later_slot,
            train_id=f"train:source:v{later_slot}",
            revision=0,
            schema_replay_proof=None,
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
    monkeypatch.setattr(durable_change_train_module, "_capture_released_schema_evidence", fake_capture)
    monkeypatch.setattr(durable_change_train_module, "_historical_schema_evidence", lambda _train: None)
    monkeypatch.setattr(
        durable_change_train_module,
        "_released_live_schema_inventory_sha256",
        lambda *_args: "inventory",
    )
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
        lambda _connection, train, **_kwargs: events.append(("verify", train.target_version)),
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
            state=DurableChangeTrainState.RELEASED,
            tier=ArchiveTier.SOURCE,
            target_version=_NEXT_SOURCE_SLOT + 1,
            schema_replay_proof=None,
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
        "_capture_released_schema_evidence",
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
    # A runtime that itself declares the slot: without it the live version is
    # simply newer than the runtime, a different (typed) refusal.
    versions = dict(ARCHIVE_VERSION_BY_TIER)
    versions[ArchiveTier.SOURCE] = _NEXT_SOURCE_SLOT
    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)

    with pytest.raises(DurableChangeTrainError, match="lacks released train evidence"):
        durable_change_train_module._reconcile_durable_change_train_startup_locked(tmp_path)


def test_fresh_archive_bootstrap_replays_no_train_and_allows_repeat_startup(tmp_path: Path) -> None:
    """The v1 baselines are the current durable schemas: fresh bootstrap releases no train."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    receipt_root = tmp_path / ".maintenance-state/durable-change-trains"
    assert durable_train_manifest_paths(receipt_root) == ()
    assert reconcile_durable_change_train_startup(tmp_path) == ()
    for tier in DURABLE_MIGRATION_ADOPTION_FLOORS:
        with closing(sqlite3.connect(tmp_path / f"{tier.value}.db")) as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == ARCHIVE_VERSION_BY_TIER[tier] == 1
    initialize_active_archive_root(tmp_path)
    assert reconcile_durable_change_train_startup(tmp_path) == ()


def test_runtime_bootstrap_refuses_an_established_archive_missing_audit(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordinary writable startup cannot create audit.db for an established archive."""
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

    with pytest.raises(RuntimeError, match="established archive is missing audit.db"):
        bootstrap.initialize_active_archive_root(archive_root)

    assert not reconciled
    assert not (archive_root / "audit.db").exists()


def test_pre_slot_source_missing_audit_is_refused_as_lost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lineage member one slot behind, without audit.db, is refused as a lost tier.

    A source tier standing below the runtime target is ordinary: the archive
    was born before that tier's numbered slot shipped. Losing ``audit.db`` on
    top of that must still produce the lost-tier refusal -- not a schema
    complaint, and never a silently recreated audit tier over durable
    evidence nobody has.

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

    with pytest.raises(RuntimeError, match="established archive is missing audit.db"):
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

    assert not (marker_root / ".bootstrap.pending").exists()
    assert durable_train_manifest_paths(marker_root) == ()
    # No train corroborates the completed bootstrap; the next ordinary startup
    # retires its receipt, since a floor-version receipt grants nothing.
    assert reconcile_durable_change_train_startup(tmp_path) == ()
    assert not (marker_root / ".bootstrap").exists()


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


@_needs_shipped_durable_slot
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
    (tmp_path / "user.db").unlink()

    with pytest.raises(RuntimeError, match="marker names a missing durable tier"):
        initialize_active_archive_root(tmp_path)
    assert not (tmp_path / "user.db").exists()


def test_bootstrap_marker_survives_index_generation_replacement(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    (tmp_path / "index.db").unlink()

    initialize_active_archive_root(tmp_path)


def test_fresh_bootstrap_archive_opens_after_its_root_is_moved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    _pin_source_runtime_version(monkeypatch, _SOURCE_ADOPTION_FLOOR)
    initialize_active_archive_root(origin)
    marker_payload = json.loads(
        (origin / ".maintenance-state" / "durable-change-trains" / ".bootstrap").read_text(encoding="utf-8")
    )
    assert "durable_identity_digest" not in marker_payload

    moved = tmp_path / "moved"
    os.rename(origin, moved)

    initialize_active_archive_root(moved)
    assert reconcile_durable_change_train_startup(moved) == ()


@_needs_shipped_durable_slot
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


@_needs_shipped_durable_slot
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


@_needs_shipped_durable_slot
def test_legacy_identity_seal_is_not_ownership_proof(tmp_path: Path) -> None:
    """A marker's legacy path-and-inode seal proves nothing (polylogue-zukcl).

    Markers written by earlier revisions carried a ``durable_identity_digest``
    over the archive root path and durable inodes. Honouring a matching seal
    short-circuited the content proof: it refused a tier that a released train
    had legitimately migrated away from its bootstrap version, and it would
    admit a transplanted marker re-sealed to its recipient. Ownership is now
    proved from durable content only, so the seal is inert.

    Anti-vacuity: restoring the seal short-circuit in
    ``_assert_fresh_durable_bootstrap_is_own`` makes the recipient open and
    inherit the donor's chain floor instead of raising.
    """
    import hashlib

    from polylogue.storage.archive_identity import ArchiveIdentity
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

    identity = ArchiveIdentity.resolve(recipient.resolve())
    payload = json.loads((donor / relative).read_text(encoding="utf-8"))
    payload["durable_identity_digest"] = hashlib.sha256(
        json.dumps(
            {"configured_root": str(identity.configured_root.absolute()), "durable_id": identity.durable_id},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    payload.pop("marker_digest", None)
    payload["marker_digest"] = durable_change_train_module._bootstrap_marker_digest(payload)
    (recipient / relative).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DurableChangeTrainError, match="not this archive's own source bootstrap evidence"):
        reconcile_durable_change_train_startup(recipient)


def test_durable_tier_ahead_of_runtime_is_refused_before_recovery(tmp_path: Path) -> None:
    """A tier above the runtime's declared version is a typed refusal (polylogue-w6nrl).

    Only a newer release can have written it, so no chain of this runtime's
    trains can admit it; startup reconciliation names the tier and both
    versions instead of reporting missing train evidence.

    Anti-vacuity: deleting the newer-than-runtime check from startup
    reconciliation surfaces the generic forward-admission error ("lacks
    released train evidence") instead of the typed refusal.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.migration_runner import DurableTierNewerThanRuntimeError

    initialize_active_archive_root(tmp_path)
    runtime_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.USER]
    with closing(sqlite3.connect(tmp_path / "user.db")) as connection:
        connection.execute(f"PRAGMA user_version = {runtime_version + 1}")
        connection.commit()

    with pytest.raises(DurableTierNewerThanRuntimeError, match="newer than this runtime supports") as refused:
        reconcile_durable_change_train_startup(tmp_path)

    assert refused.value.tier is ArchiveTier.USER
    assert (refused.value.live_version, refused.value.runtime_version) == (runtime_version + 1, runtime_version)


def test_newer_release_archive_is_refused_by_version_before_released_schema_proof(tmp_path: Path) -> None:
    """Newer live schema is typed version skew before current schema admission."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.migration_runner import DurableTierNewerThanRuntimeError

    initialize_active_archive_root(tmp_path)
    newer = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] + 1
    with closing(sqlite3.connect(tmp_path / "source.db")) as connection:
        connection.execute("CREATE TABLE newer_release_additive_table (id INTEGER PRIMARY KEY)")
        connection.execute(f"PRAGMA user_version = {newer}")
        connection.commit()
    with pytest.raises(DurableTierNewerThanRuntimeError, match="newer than this runtime supports"):
        reconcile_durable_change_train_startup(tmp_path)


@_needs_shipped_durable_slot
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


def test_fresh_bootstrap_receipt_rejects_recorded_version_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    _pin_source_runtime_version(monkeypatch, _SOURCE_ADOPTION_FLOOR)
    initialize_active_archive_root(tmp_path)
    marker = tmp_path / ".maintenance-state" / "durable-change-trains" / ".bootstrap"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    cast(dict[str, int], payload["versions"])["source"] += 1
    marker.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DurableChangeTrainError, match="marker digest mismatch"):
        reconcile_durable_change_train_startup(tmp_path)


def test_source_train_identity_survives_late_user_tier_initialization(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.write_lease import write_lease

    source_path = tmp_path / "source.db"
    with write_lease("test.source-baseline-bootstrap", archive_root=tmp_path):
        initialize_archive_database(source_path, ArchiveTier.SOURCE, expected_version=1)
    with sqlite3.connect(source_path) as conn:
        before = migration_runner.capture_durable_database_evidence(conn, ArchiveTier.SOURCE)

    with write_lease("test.late-user-baseline-bootstrap", archive_root=tmp_path):
        initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER, expected_version=1)
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


def test_admission_rejects_stale_current_and_target_versions() -> None:
    train = _declared(ArchiveTier.SOURCE)
    with pytest.raises(DurableChangeTrainError, match="stale durable train current"):
        _admit_for_test(
            train,
            observed_current_version=0,
            schema_replay_proof=_parity(ArchiveTier.SOURCE),
            admission_evidence_ref="proof:admit",
            migration_claims=(train.migration,),
            canonical_target_version=_TARGET_VERSION,
        )
    with pytest.raises(DurableChangeTrainError, match="stale durable train target"):
        _admit_for_test(
            train,
            observed_current_version=_CURRENT_VERSION,
            schema_replay_proof=_parity(ArchiveTier.SOURCE),
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
    parity = _parity(ArchiveTier.SOURCE, claim=first)
    with pytest.raises(DurableChangeTrainError, match="collision.*rebase/renumber") as exc_info:
        _admit_for_test(
            train,
            observed_current_version=slot - 1,
            schema_replay_proof=parity,
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
        _admit_for_test(
            duplicate,
            observed_current_version=_CURRENT_VERSION,
            schema_replay_proof=_parity(ArchiveTier.SOURCE),
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


def test_schema_replay_proof_mismatch_blocks_admission() -> None:
    mismatch = _parity(ArchiveTier.SOURCE, include_durable_items=False, matches=False)
    assert mismatch.matches is False
    train = _declared(ArchiveTier.SOURCE)
    with pytest.raises(DurableChangeTrainError, match="schema replay"):
        _admit_for_test(
            train,
            observed_current_version=_CURRENT_VERSION,
            schema_replay_proof=mismatch,
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
    # Schema-only rehearsal accepts the index, but two existing values violate
    # it on the live archive. That exercises data-dependent transaction failure.
    failing_sql = _DATA_DEPENDENT_FAILURE_SQL
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    claim = _claim(ArchiveTier.SOURCE, failing_sql)
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE, sql=failing_sql)
    train = _admitted(ArchiveTier.SOURCE, claim=claim)
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO base_items VALUES ('base-2', 'preserve-me')")
        conn.commit()
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
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE)
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
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE)
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
    # Schema-only rehearsal accepts the index, but two existing values violate
    # it on the live archive. That exercises data-dependent transaction failure.
    failing_sql = _DATA_DEPENDENT_FAILURE_SQL
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE, sql=failing_sql)
    train = _admitted(ArchiveTier.SOURCE, claim=_claim(ArchiveTier.SOURCE, failing_sql))
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO base_items VALUES ('base-2', 'preserve-me')")
        conn.commit()
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
    # Schema-only rehearsal accepts the index, but two existing values violate
    # it on the live archive. That exercises data-dependent transaction failure.
    failing_sql = _DATA_DEPENDENT_FAILURE_SQL
    db_path = tmp_path / "source.db"
    _create_current_database(db_path)
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE, sql=failing_sql)
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO base_items VALUES ('base-2', 'preserve-me')")
        conn.commit()
        train = _reserve_and_authorize(
            conn,
            _admitted(ArchiveTier.SOURCE, claim=_claim(ArchiveTier.SOURCE, failing_sql)),
            archive_root=tmp_path,
        )
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
    _install_synthetic_migration(tmp_path, monkeypatch, ArchiveTier.SOURCE)
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
        actual_parity = train.schema_replay_proof
        assert actual_parity is not None
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
            schema_replay_proof=actual_parity,
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
        parity = released.schema_replay_proof
        assert parity is not None
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
            schema_replay_proof=parity,
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
    parity = payload["schema_replay_proof"]
    assert isinstance(parity, dict)
    parity["matches"] = False
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256")
    payload["manifest_sha256"] = migration_runner._canonical_json_sha256(unsigned)
    path = tmp_path / "forged-train.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DurableChangeTrainError, match="schema replay terminal schema"):
        load_durable_change_train_manifest(path)


@pytest.mark.parametrize(
    ("drop_sql", "accepted"),
    [
        ("DROP INDEX items_kind; CREATE UNIQUE INDEX items_kind ON items(kind) WHERE kind IN ('new');", True),
        ("DROP INDEX items_kind;", False),
        ("DROP INDEX items_kind; CREATE INDEX other_kind ON items(kind) WHERE kind IN ('new');", False),
        ("DROP TABLE items;", False),
        ("DROP VIEW item_view;", False),
        ("DROP TRIGGER item_trigger;", False),
        ("ALTER TABLE items DROP COLUMN kind;", False),
        ('ALTER TABLE "items" DROP kind;', False),
        ("ALTER/* boundary */TABLE items DROP COLUMN kind;", False),
        (
            "ALTER TABLE items DROP COLUMN kind; DROP INDEX items_kind; CREATE INDEX items_kind ON items(kind) WHERE kind IN ('new');",
            False,
        ),
    ],
)
def test_backup_required_mixed_train_classifies_each_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drop_sql: str, accepted: bool
) -> None:
    package = f"fixture_mixed_drop_{tmp_path.name.replace('-', '_')}"
    directory = tmp_path / package / "source"
    directory.mkdir(parents=True)
    (directory.parent / "__init__.py").write_text("")
    (directory / "__init__.py").write_text("")
    sql = "CREATE TABLE receipts(id TEXT PRIMARY KEY) STRICT; ALTER TABLE items ADD COLUMN receipt_id TEXT; " + drop_sql
    name = "002_mixed_index_replacement.sql"
    claim = durable_migration_claim_for_sql(ArchiveTier.SOURCE, name, sql, owner_ref="owner:mixed-drops")
    assert claim.requires_backup
    train = declare_durable_change_train(
        train_id="source-mixed-index-replacement",
        tier=ArchiveTier.SOURCE,
        current_version=1,
        target_version=2,
        slot=2,
        owner_ref="owner:mixed-drops",
        migration=claim,
        riders=(_production_rider(),),
        backup_plan_ref="proof:mixed-train-backup",
        declared_at_ms=1,
    )
    (directory / "002.train.json").write_text(json.dumps(migration_runner.durable_change_train_to_payload(train)))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(durable_change_train_module, "_migration_package", lambda _tier: f"{package}.source")
    if accepted:
        result = validate_durable_migration_sidecars(ArchiveTier.SOURCE, ((name, sql),))
        assert result[0].train.migration.requires_backup
        with pytest.raises(migration_runner.MigrationError):
            durable_migration_claim_for_sql(
                ArchiveTier.SOURCE,
                name,
                "-- migration-safety: row-preserving-index-replacement\n" + sql,
            )
    else:
        with pytest.raises(DurableChangeTrainError, match="unapproved drop"):
            validate_durable_migration_sidecars(ArchiveTier.SOURCE, ((name, sql),))


@pytest.mark.parametrize("probe_kind", ["source", "user", "user-file"])
def test_runtime_consumer_probe_keeps_native_owner_until_creator_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe_kind: str
) -> None:
    from polylogue.storage.sqlite import connection_profile, managed_connection
    from tests.infra.sqlite_cursor_settlement import ControlledConnection

    def connect(database: str | Path, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = sqlite3.connect(database, *args, factory=ControlledConnection, **kwargs)
        assert isinstance(connection, ControlledConnection)
        return connection

    monkeypatch.setattr(managed_connection, "connect_measured", connect)
    monkeypatch.setattr(connection_profile, "connect_measured", connect)
    helper = {
        "source": durable_change_train_module._runtime_probe_source_connection,
        "user": durable_change_train_module._runtime_probe_user_connection,
        "user-file": durable_change_train_module._runtime_probe_user_file_connection,
    }[probe_kind]
    connection: ControlledConnection | None = None
    with pytest.raises(connection_profile.NativeConnectionSettlementError) as failed:
        with helper() as actual:
            assert isinstance(actual, ControlledConnection)
            connection = actual
            actual.close_failure = OSError("synthetic probe close remains unsettled")
    assert connection is not None
    owner = failed.value.owner
    directory = Path(owner.scratch_directory.name) if owner.scratch_directory is not None else None
    try:
        assert connection.close_attempts == 1
        if probe_kind == "user-file":
            assert directory is not None and (directory / "user.db").is_file()
        assert owner.connection is connection
        with pytest.raises(RuntimeError, match="terminal cleanup"):
            owner.require_connection()
    finally:
        connection.close_failure = None
        owner.close()
    assert connection.close_attempts == 2
    if directory is not None:
        assert not directory.exists()


def test_canonical_train_inventory_retains_actual_connection_on_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite import connection_profile, managed_connection
    from tests.infra.sqlite_cursor_settlement import ControlledConnection

    def connect(database: str | Path, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = sqlite3.connect(database, *args, factory=ControlledConnection, **kwargs)
        assert isinstance(connection, ControlledConnection)
        return connection

    capture = migration_runner.capture_durable_schema_inventory
    actual: ControlledConnection | None = None

    def capture_and_arm(connection: sqlite3.Connection) -> object:
        nonlocal actual
        result = capture(connection)
        assert isinstance(connection, ControlledConnection)
        actual = connection
        connection.close_failure = OSError("synthetic canonical inventory close remains unsettled")
        return result

    monkeypatch.setattr(managed_connection, "connect_measured", connect)
    monkeypatch.setattr(migration_runner, "capture_durable_schema_inventory", capture_and_arm)
    canonical = durable_change_train_module._canonical_schema_inventory_for_ddl
    canonical.cache_clear()
    with pytest.raises(connection_profile.NativeConnectionSettlementError) as failed:
        canonical(ArchiveTier.USER, 1, ARCHIVE_BASELINE_DDL_BY_TIER[ArchiveTier.USER], ())
    assert actual is not None
    try:
        assert actual.close_attempts == 1
        assert failed.value.owner.connection is actual
        with pytest.raises(RuntimeError, match="terminal cleanup"):
            failed.value.owner.require_connection()
    finally:
        actual.close_failure = None
        failed.value.owner.close()
        canonical.cache_clear()
    assert actual.close_attempts == 2


def test_raw_failure_probe_uses_authenticated_train_snapshot_without_admitting_file_skew(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle
    from polylogue.storage.sqlite.archive_tiers.bootstrap import (
        RuntimeTierProbeAuthority,
        initialize_runtime_tier_probe,
        runtime_tier_probe_authority,
    )
    from polylogue.storage.sqlite.managed_connection import sqlite_connection

    # The runtime declares one step past the floor, so the authenticated
    # floor snapshot is a version the ordinary reader must not admit.
    monkeypatch.setitem(ARCHIVE_VERSION_BY_TIER, ArchiveTier.SOURCE, _SOURCE_ADOPTION_FLOOR + 1)
    inventory = durable_change_train_module._canonical_schema_inventory(ArchiveTier.SOURCE, _SOURCE_ADOPTION_FLOOR)
    path = tmp_path / "source.db"
    authority = RuntimeTierProbeAuthority(ArchiveTier.SOURCE, _SOURCE_ADOPTION_FLOOR, inventory.sha256)
    observed_versions: list[int] = []

    def reader(source_path: Path | None, *, sample_limit: int, _connection: sqlite3.Connection) -> object:
        assert source_path is None
        assert _connection.in_transaction
        assert _connection.execute("PRAGMA query_only").fetchone()[0] == 1
        observed_versions.append(int(_connection.execute("PRAGMA user_version").fetchone()[0]))
        snapshot = read_raw_failure_lifecycle(source_path, sample_limit=sample_limit, _connection=_connection)
        assert snapshot.healthy
        return snapshot

    with runtime_tier_probe_authority(authority):
        with sqlite_connection(path) as connection:
            initialize_runtime_tier_probe(connection, ArchiveTier.SOURCE, probe_path=path)
        durable_change_train_module._probe_raw_failure_lifecycle(reader, tmp_path)
    assert observed_versions == [_SOURCE_ADOPTION_FLOOR]
    ordinary = read_raw_failure_lifecycle(path)
    assert not ordinary.available
    assert ordinary.state == "unavailable"


@pytest.mark.parametrize("cancelled", [False, True])
def test_runtime_probe_directory_survives_native_close_until_creator_retry(
    monkeypatch: pytest.MonkeyPatch, cancelled: bool
) -> None:
    from polylogue.storage.sqlite import connection_profile, managed_connection
    from tests.infra.sqlite_cursor_settlement import ControlledConnection

    def connect(database: str | Path, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = sqlite3.connect(database, *args, factory=ControlledConnection, **kwargs)
        assert isinstance(connection, ControlledConnection)
        return connection

    monkeypatch.setattr(managed_connection, "connect_measured", connect)
    directory: Path | None = None
    cancellation = InterruptedError("synthetic probe cancellation")
    with pytest.raises(connection_profile.NativeConnectionSettlementError) as failed:
        with durable_change_train_module._runtime_probe_directory(prefix="polylogue-train-lifetime-test-") as scratch:
            directory = scratch
            with managed_connection.sqlite_connection(scratch / "probe.db") as connection:
                assert isinstance(connection, ControlledConnection)
                connection.execute("CREATE TABLE proof(value TEXT)")
                connection.close_failure = OSError("synthetic probe close remains unsettled")
                if cancelled:
                    raise cancellation
    import gc

    owner = failed.value.owner
    actual = owner.connection
    assert isinstance(actual, ControlledConnection)
    if cancelled:
        assert failed.value.__cause__ is cancellation
    actual.close_failure = None
    del failed
    cancellation.__traceback__ = None
    gc.collect()
    try:
        assert directory is not None and (directory / "probe.db").is_file()
        assert actual.close_attempts == 1
    finally:
        owner.close()
    assert actual.close_attempts == 2
    gc.collect()
    assert not directory.exists()

"""Durable source/user migration change-train authority."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Final, cast

from polylogue.storage.sqlite import migration_runner as _migration_runner
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import (
    DURABLE_CHANGE_TRAIN_FORMAT,
    DurableChangeTrain,
    DurableChangeTrainApplyError,
    DurableChangeTrainError,
    DurableChangeTrainRecoveryError,
    DurableChangeTrainState,
    DurableFreshDDLParityProof,
    DurableMigrationClaim,
    DurableRuntimeConsumerResult,
    MigrationResult,
    _assert_durable_database_continuity,
    _validate_riders,
    add_durable_change_train_rider,
    admit_durable_change_train,
    apply_durable_change_train,
    authorize_durable_change_train_backup,
    capture_durable_database_evidence,
    declare_durable_change_train,
    durable_change_train_from_payload,
    durable_change_train_to_payload,
    durable_migration_claim_for_sql,
    durable_migration_claims,
    durable_migration_collision_report,
    find_durable_migration_collisions,
    load_durable_change_train_manifest,
    prove_durable_change_train,
    prove_durable_fresh_ddl_parity,
    reconcile_interrupted_durable_change_train,
    record_durable_writer_release,
    recover_durable_change_train,
    release_durable_change_train,
    reserve_durable_change_train,
    validate_durable_change_train_manifest,
    write_durable_change_train_manifest,
)

DURABLE_MIGRATION_ADOPTION_FLOORS: Final[dict[ArchiveTier, int]] = {
    ArchiveTier.SOURCE: 26,
    ArchiveTier.USER: 10,
}
_SIDECAR_NAME_RE = re.compile(r"^(?P<slot>\d{3,})\.train\.json$")
_MIGRATION_NAME_RE = re.compile(r"^(?P<slot>\d{3,})_[a-z0-9_]+\.sql$")
_DROP_SQL_RE = re.compile(r"(?is)\bDROP\s+(?:TABLE|INDEX|TRIGGER|VIEW)\b")


@dataclass(frozen=True, slots=True)
class DurableMigrationSidecar:
    """A deterministic package resource binding one SQL slot to its train."""

    tier: ArchiveTier
    slot: int
    resource_name: str
    train: DurableChangeTrain


@dataclass(frozen=True, slots=True)
class DurableChangeTrainExecution:
    """Result of one production durable change-train execution."""

    train: DurableChangeTrain | None
    manifest_path: Path | None
    migration_result: MigrationResult | None


def durable_migration_sidecar_name(slot: int) -> str:
    """Return the only accepted Git path for a numbered train sidecar."""
    if slot < 1:
        raise DurableChangeTrainError(f"durable migration sidecar slot must be positive: {slot}")
    return f"{slot:03d}.train.json"


def _migration_package(tier: ArchiveTier) -> str:
    return f"polylogue.storage.sqlite.migrations.{tier.value}"


def _sidecar_slot(name: str) -> int | None:
    match = _SIDECAR_NAME_RE.fullmatch(name)
    return int(match.group("slot")) if match is not None else None


def _load_sidecar_resource(tier: ArchiveTier, resource_name: str) -> DurableMigrationSidecar:
    try:
        resource = resources.files(_migration_package(tier)).joinpath(resource_name)
        raw = json.loads(resource.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DurableChangeTrainError(
            f"missing durable migration train sidecar for {tier.value}: {resource_name}"
        ) from exc
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise DurableChangeTrainError(
            f"malformed durable migration train sidecar for {tier.value}: {resource_name}"
        ) from exc
    if not isinstance(raw, dict):
        raise DurableChangeTrainError(f"durable migration train sidecar must be an object: {resource_name}")
    train = durable_change_train_from_payload(raw)
    slot = _sidecar_slot(resource_name)
    if slot is None:
        raise DurableChangeTrainError(f"invalid durable migration train sidecar name: {resource_name}")
    return DurableMigrationSidecar(tier=tier, slot=slot, resource_name=resource_name, train=train)


def _validate_sidecar_binding(
    sidecar: DurableMigrationSidecar,
    *,
    migration_name: str,
    sql: str,
) -> None:
    train = sidecar.train
    expected_claim = durable_migration_claim_for_sql(
        sidecar.tier,
        migration_name,
        sql,
        owner_ref=train.migration.owner_ref,
    )
    if train.state is not DurableChangeTrainState.DECLARED:
        raise DurableChangeTrainError(f"durable migration sidecar must begin declared: {sidecar.resource_name}")
    if train.tier is not sidecar.tier or train.current_version != sidecar.slot - 1:
        raise DurableChangeTrainError(
            f"durable migration sidecar version is stale or mismatched: {sidecar.resource_name}"
        )
    if train.target_version != sidecar.slot or train.slot != sidecar.slot:
        raise DurableChangeTrainError(f"durable migration sidecar target/slot mismatch: {sidecar.resource_name}")
    if Path(train.migration.path).name != migration_name:
        raise DurableChangeTrainError(f"durable migration sidecar SQL filename mismatch: {sidecar.resource_name}")
    if train.migration.sql_sha256 != expected_claim.sql_sha256:
        raise DurableChangeTrainError(f"durable migration sidecar SQL SHA-256 mismatch: {sidecar.resource_name}")
    if train.migration.requires_backup != expected_claim.requires_backup:
        raise DurableChangeTrainError(f"durable migration sidecar backup policy mismatch: {sidecar.resource_name}")
    _validate_riders(train)
    if train.migration.requires_backup and not train.backup_plan_ref:
        raise DurableChangeTrainError(
            f"backup-required durable migration sidecar lacks a backup plan: {sidecar.resource_name}"
        )
    if _DROP_SQL_RE.search(sql) is not None and not train.drop_constraints:
        raise DurableChangeTrainError(f"durable migration sidecar forbids an unapproved drop: {sidecar.resource_name}")


def validate_durable_migration_sidecars(
    tier: ArchiveTier,
    migrations: Sequence[tuple[str, str]],
) -> tuple[DurableMigrationSidecar, ...]:
    """Require and validate every post-floor SQL slot's checked-in sidecar.

    Discovery uses ``importlib.resources`` so the policy follows the package
    resources consumed by production, including installed wheels. Extra,
    malformed, stale, or orphaned sidecars are rejected as well.
    """
    if tier not in DURABLE_MIGRATION_ADOPTION_FLOORS:
        return ()
    by_slot: dict[int, tuple[str, str]] = {}
    for name, sql in migrations:
        match = _MIGRATION_NAME_RE.fullmatch(name)
        if match is None:
            continue
        slot = int(match.group("slot"))
        if slot in by_slot:
            raise DurableChangeTrainError(f"duplicate durable migration slot: {tier.value}/{slot:03d}")
        by_slot[slot] = (name, sql)
    floor = DURABLE_MIGRATION_ADOPTION_FLOORS[tier]
    try:
        package = resources.files(_migration_package(tier))
        sidecar_names = {item.name for item in package.iterdir() if item.name.endswith(".train.json")}
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        if any(slot > floor for slot in by_slot):
            raise DurableChangeTrainError(f"cannot discover durable migration train sidecars for {tier.value}") from exc
        return ()
    observed: list[DurableMigrationSidecar] = []
    for name in sorted(sidecar_names):
        sidecar_slot = _sidecar_slot(name)
        if sidecar_slot is None:
            raise DurableChangeTrainError(f"invalid durable migration train sidecar name: {name}")
        if sidecar_slot <= floor:
            raise DurableChangeTrainError(
                f"durable migration train sidecar is below adoption floor: {tier.value}/{name}"
            )
        if sidecar_slot not in by_slot:
            raise DurableChangeTrainError(
                f"durable migration train sidecar has no matching SQL resource: {tier.value}/{name}"
            )
        sidecar = _load_sidecar_resource(tier, name)
        _validate_sidecar_binding(
            sidecar,
            migration_name=by_slot[sidecar_slot][0],
            sql=by_slot[sidecar_slot][1],
        )
        observed.append(sidecar)
    for slot, (name, sql) in sorted(by_slot.items()):
        if slot <= floor:
            continue
        expected_name = durable_migration_sidecar_name(slot)
        if expected_name not in sidecar_names:
            raise DurableChangeTrainError(f"missing durable migration train sidecar: {tier.value}/{expected_name}")
        sidecar = _load_sidecar_resource(tier, expected_name)
        _validate_sidecar_binding(sidecar, migration_name=name, sql=sql)
        if sidecar.slot != slot:
            raise DurableChangeTrainError(f"durable migration train sidecar slot mismatch: {expected_name}")
    expected_slots = tuple(range(floor + 1, max(by_slot, default=floor) + 1))
    observed_slots = tuple(sorted(slot for slot in by_slot if slot > floor))
    if observed_slots != expected_slots:
        raise DurableChangeTrainError(
            f"durable migration train sidecars are noncontiguous for {tier.value}: "
            f"expected {expected_slots}, found {observed_slots}"
        )
    return tuple(observed)


def durable_change_train_policy_report(tier: ArchiveTier) -> dict[str, object]:
    """Emit reservations and every discovered violation for schema policy JSON."""
    reservations: list[dict[str, object]] = []
    violations: list[str] = []
    try:
        package = resources.files(_migration_package(tier))
        migrations = tuple(
            (item.name, item.read_text(encoding="utf-8"))
            for item in package.iterdir()
            if _MIGRATION_NAME_RE.fullmatch(item.name) is not None
        )
        sidecars = validate_durable_migration_sidecars(tier, migrations)
        for sidecar in sidecars:
            reservation = sidecar.train.reservation
            if reservation is not None:
                reservations.append(
                    {
                        "tier": tier.value,
                        "slot": sidecar.slot,
                        "resource": sidecar.resource_name,
                        "reservation": {
                            "reservation_id": reservation.reservation_id,
                            "owner_ref": reservation.owner_ref,
                            "archive_root": reservation.archive_root,
                            "tier_path": reservation.tier_path,
                            "active": reservation.active,
                        },
                    }
                )
    except (DurableChangeTrainError, ModuleNotFoundError, FileNotFoundError, OSError) as exc:
        violations.append(str(exc))
    return {
        "tier": tier.value,
        "adoption_floor": DURABLE_MIGRATION_ADOPTION_FLOORS.get(tier),
        "reservations": reservations,
        "violations": violations,
        "ok": not violations,
    }


def durable_change_train_manifest_path(archive_root: Path, tier: ArchiveTier, slot: int) -> Path:
    """Return the stable persisted authority path for one archive train."""
    if tier not in DURABLE_MIGRATION_ADOPTION_FLOORS:
        raise DurableChangeTrainError(f"{tier.value} has no durable change-train authority")
    if slot <= DURABLE_MIGRATION_ADOPTION_FLOORS[tier]:
        raise DurableChangeTrainError(f"durable train slot is below the adoption floor: {tier.value}/{slot}")
    return archive_root / ".maintenance-state" / "durable-change-trains" / f"{tier.value}-{slot:03d}.json"


def durable_migration_sidecar_for_slot(tier: ArchiveTier, slot: int) -> DurableMigrationSidecar | None:
    """Load the package sidecar for the next numbered production migration."""
    if tier not in DURABLE_MIGRATION_ADOPTION_FLOORS:
        return None
    steps = _migration_runner._load_migrations(tier)
    step = next((item for item in steps if item.version == slot), None)
    if step is None:
        return None
    sidecars = validate_durable_migration_sidecars(tier, tuple((item.name, item.sql) for item in steps))
    return next((item for item in sidecars if item.slot == slot), None)


def _persist_train_transition(path: Path, train: DurableChangeTrain, *, expected_revision: int) -> DurableChangeTrain:
    write_durable_change_train_manifest(path, train, expected_revision=expected_revision)
    return load_durable_change_train_manifest(path)


def _fresh_ddl_parity_for_train(
    train: DurableChangeTrain,
    *,
    migrated_connection: sqlite3.Connection | None = None,
) -> DurableFreshDDLParityProof:
    """Compare a live result, or two canonical creates, against bootstrap DDL."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier

    if migrated_connection is None:
        with sqlite3.connect(":memory:") as migrated, sqlite3.connect(":memory:") as fresh:
            initialize_archive_tier(migrated, train.tier)
            initialize_archive_tier(fresh, train.tier)
            return prove_durable_fresh_ddl_parity(
                train.tier,
                train.target_version,
                migrated_connection=migrated,
                fresh_connection=fresh,
                evidence_ref=f"proof:canonical-bootstrap:{train.tier.value}:v{train.target_version}",
            )
    with sqlite3.connect(":memory:") as fresh:
        initialize_archive_tier(fresh, train.tier)
        return prove_durable_fresh_ddl_parity(
            train.tier,
            train.target_version,
            migrated_connection=migrated_connection,
            fresh_connection=fresh,
            evidence_ref=f"proof:recovered-bootstrap:{train.tier.value}:v{train.target_version}",
        )


def _runtime_consumer_results(
    train: DurableChangeTrain,
    archive_root: Path,
) -> tuple[DurableRuntimeConsumerResult, ...]:
    """Invoke each declared production probe before recording behavior proof."""
    results: list[DurableRuntimeConsumerResult] = []
    for rider in train.riders:
        for consumer in rider.runtime_consumers:
            reference = consumer.production_ref
            module_ref, separator, symbol_ref = reference.partition(":")
            if not separator or not symbol_ref:
                raise DurableChangeTrainError(f"runtime consumer reference is not importable: {reference}")
            module_name = module_ref.removesuffix(".py").replace("/", ".")
            try:
                value: object = importlib.import_module(module_name)
                for component in symbol_ref.split("."):
                    value = getattr(value, component)
            except (ImportError, AttributeError) as exc:
                raise DurableChangeTrainError(
                    f"runtime consumer {consumer.consumer_id} cannot resolve production reference {reference}"
                ) from exc
            if not callable(value):
                raise DurableChangeTrainError(f"runtime consumer {consumer.consumer_id} is not callable: {reference}")
            try:
                if reference.endswith(":initialize_archive_database"):
                    value(archive_root / f"{train.tier.value}.db", train.tier, allow_create=False)
                elif reference.endswith(":initialize_archive_tier"):
                    with sqlite3.connect(":memory:") as probe:
                        value(probe, train.tier)
                elif not any(
                    parameter.default is inspect.Parameter.empty
                    and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    for parameter in inspect.signature(value).parameters.values()
                ):
                    value()
                else:
                    raise DurableChangeTrainError(
                        f"runtime consumer {consumer.consumer_id} has no durable probe adapter: {reference}"
                    )
            except DurableChangeTrainError:
                raise
            except Exception as exc:
                raise DurableChangeTrainError(
                    f"runtime consumer {consumer.consumer_id} probe failed: {reference}: {exc}"
                ) from exc
            results.append(
                DurableRuntimeConsumerResult(
                    consumer_id=consumer.consumer_id,
                    behavior_proof_ref=consumer.behavior_proof_ref,
                    passed=True,
                    detail=f"resolved {reference}",
                )
            )
    return tuple(results)


def _open_existing_tier(tier_path: Path) -> sqlite3.Connection:
    """Open an existing durable tier without allowing SQLite to create it."""
    try:
        metadata = tier_path.lstat()
    except FileNotFoundError as exc:
        raise DurableChangeTrainError(
            "durable tier is missing; refusing startup initialization/release until restored"
        ) from exc
    if tier_path.is_symlink() or not tier_path.is_file() or metadata.st_nlink != 1:
        raise DurableChangeTrainError(
            "durable tier was replaced by an unsafe file; refusing startup initialization/release"
        )
    try:
        return sqlite3.connect(f"{tier_path.resolve(strict=True).as_uri()}?mode=rw", uri=True)
    except (OSError, sqlite3.Error) as exc:
        raise DurableChangeTrainError("durable tier could not be opened without initialization") from exc


def _verify_persisted_live_tier_continuity(conn: sqlite3.Connection, train: DurableChangeTrain) -> None:
    """Prove the exact reopened connection still names the persisted durable tier."""
    if train.apply_evidence is None:
        raise DurableChangeTrainError(f"{train.state.value} train lacks post-apply continuity evidence")
    actual = capture_durable_database_evidence(conn, train.tier)
    expected = train.apply_evidence.post
    if actual.user_version != train.target_version:
        raise DurableChangeTrainError(
            f"{train.tier.value} durable tier continuity proof failed; refusing startup initialization/release"
        )
    try:
        _assert_durable_database_continuity(actual, expected, label=train.tier.value)
    except DurableChangeTrainError as exc:
        raise DurableChangeTrainError(
            f"{train.tier.value} durable tier continuity proof failed; refusing startup initialization/release"
        ) from exc


def _prove_and_release_persisted_train(
    archive_root: Path,
    manifest_path: Path,
    train: DurableChangeTrain,
    *,
    runtime_consumer_results: Sequence[DurableRuntimeConsumerResult] | None = None,
) -> DurableChangeTrain:
    """Finish a persisted applied/proven train after an interrupted process."""
    tier_path = archive_root / f"{train.tier.value}.db"
    if train.state is DurableChangeTrainState.APPLIED:
        live = _open_existing_tier(tier_path)
        try:
            _verify_persisted_live_tier_continuity(live, train)
            if train.reservation is not None and train.reservation.active:
                previous_revision = train.revision
                train = record_durable_writer_release(
                    train,
                    evidence_ref=f"proof:startup-writer-release:{train.train_id}",
                )
                train = _persist_train_transition(manifest_path, train, expected_revision=previous_revision)
            actual_parity = _fresh_ddl_parity_for_train(train, migrated_connection=live)
            runtime_results = (
                tuple(runtime_consumer_results)
                if runtime_consumer_results is not None
                else _runtime_consumer_results(train, archive_root)
            )
            restart = _migration_runner.capture_durable_restart_convergence(
                live,
                train,
                runtime_consumers=runtime_results,
                evidence_ref=f"proof:startup-restart:{train.train_id}",
            )
            previous_revision = train.revision
            train = prove_durable_change_train(
                train,
                fresh_ddl_parity=actual_parity,
                runtime_consumers=runtime_results,
                restart_convergence=restart,
                proof_refs=(f"proof:startup-recovery:{train.train_id}",),
            )
            _verify_persisted_live_tier_continuity(live, train)
            train = _persist_train_transition(manifest_path, train, expected_revision=previous_revision)
        finally:
            live.close()
    if train.state is DurableChangeTrainState.PROVEN:
        with _open_existing_tier(tier_path) as live:
            _verify_persisted_live_tier_continuity(live, train)
            previous_revision = train.revision
            train = release_durable_change_train(
                train,
                evidence_ref=f"proof:startup-train-release:{train.train_id}",
            )
            train = _persist_train_transition(manifest_path, train, expected_revision=previous_revision)
    return train


def execute_durable_change_train(
    archive_root: Path,
    tier: ArchiveTier,
    *,
    backup_manifest: Path | None,
    daemon_stopped_evidence_ref: str,
    single_writer_evidence_ref: str,
    runtime_consumer_results: Sequence[DurableRuntimeConsumerResult] | None = None,
    release_archive_ownership: Callable[[], None],
) -> DurableChangeTrainExecution:
    """Execute the real maintenance route through every persisted train state."""
    reconcile_durable_change_train_startup(archive_root)
    tier_path = archive_root / f"{tier.value}.db"
    with _open_existing_tier(tier_path) as probe:
        current_version = int(probe.execute("PRAGMA user_version").fetchone()[0] or 0)
    runtime_target_version = cast(dict[ArchiveTier, int], vars(_migration_runner)["ARCHIVE_VERSION_BY_TIER"])[tier]
    if current_version > runtime_target_version:
        raise DurableChangeTrainError(
            f"{tier.value} tier version {current_version} is newer than runtime target {runtime_target_version}"
        )
    if current_version < runtime_target_version:
        # Validate the complete route before any historical step can commit.
        # This prevents an old archive from being advanced to the adoption
        # floor when a later SQL/sidecar slot is missing.
        migration_steps = _migration_runner._load_migrations(tier)
        validate_durable_migration_sidecars(
            tier,
            tuple((step.name, step.sql) for step in migration_steps),
        )
        with sqlite3.connect(":memory:") as preflight:
            _migration_runner._pending_migration_steps(
                preflight,
                tier,
                current_version=current_version,
                target_version=runtime_target_version,
            )
    legacy_result: MigrationResult | None = None
    floor = DURABLE_MIGRATION_ADOPTION_FLOORS.get(tier)
    if floor is not None and current_version < floor:
        with sqlite3.connect(tier_path) as conn:
            legacy_result = _migration_runner.migrate_archive_tier(
                conn,
                tier,
                backup_manifest=backup_manifest,
                target_version=floor,
            )
        current_version = legacy_result.to_version
    sidecar = durable_migration_sidecar_for_slot(tier, current_version + 1)
    if sidecar is None:
        if current_version != runtime_target_version:
            raise DurableChangeTrainError(
                f"durable migration chain for {tier.value} stops at v{current_version}; "
                f"runtime requires v{runtime_target_version} and the next train sidecar is missing"
            )
        return DurableChangeTrainExecution(train=None, manifest_path=None, migration_result=legacy_result)

    manifest_path = durable_change_train_manifest_path(archive_root, tier, sidecar.slot)
    if manifest_path.exists():
        train = load_durable_change_train_manifest(manifest_path)
        if train.train_id != sidecar.train.train_id or train.migration != sidecar.train.migration:
            raise DurableChangeTrainError(f"persisted durable train does not match package sidecar: {manifest_path}")
    else:
        train = _persist_train_transition(manifest_path, sidecar.train, expected_revision=-1)

    if train.state is DurableChangeTrainState.RELEASED:
        with _open_existing_tier(tier_path) as live:
            live_version = int(live.execute("PRAGMA user_version").fetchone()[0] or 0)
            if live_version != runtime_target_version:
                raise DurableChangeTrainError(
                    f"released {tier.value} train {train.train_id} expects live v{runtime_target_version}, "
                    f"found v{live_version}; authorize a new execution"
                )
            _verify_persisted_live_tier_continuity(live, train)
        return DurableChangeTrainExecution(train=train, manifest_path=manifest_path, migration_result=None)

    if train.state is DurableChangeTrainState.DECLARED:
        previous_revision = train.revision
        train = admit_durable_change_train(
            train,
            observed_current_version=current_version,
            fresh_ddl_parity=_fresh_ddl_parity_for_train(train),
            admission_evidence_ref=f"proof:maintenance-admission:{train.train_id}",
            migration_claims=(sidecar.train.migration,),
            canonical_target_version=cast(dict[ArchiveTier, int], vars(_migration_runner)["ARCHIVE_VERSION_BY_TIER"])[
                tier
            ],
        )
        train = _persist_train_transition(manifest_path, train, expected_revision=previous_revision)
    if train.state is DurableChangeTrainState.ADMITTED:
        previous_revision = train.revision
        train = reserve_durable_change_train(
            train,
            reservation_id=f"maintenance:{train.train_id}",
            reservation_owner_ref=train.owner_ref,
            archive_root=archive_root,
            tier_path=tier_path,
            daemon_stopped_evidence_ref=daemon_stopped_evidence_ref,
            single_writer_evidence_ref=single_writer_evidence_ref,
        )
        train = _persist_train_transition(manifest_path, train, expected_revision=previous_revision)
    if train.state is DurableChangeTrainState.RESERVED:
        previous_revision = train.revision
        with sqlite3.connect(tier_path) as conn:
            train = authorize_durable_change_train_backup(
                conn,
                train,
                backup_manifest=backup_manifest,
                evidence_ref=f"proof:maintenance-backup:{train.train_id}",
            )
        train = _persist_train_transition(manifest_path, train, expected_revision=previous_revision)
    if train.state is DurableChangeTrainState.BACKUP_AUTHORIZED:
        previous_revision = train.revision
        try:
            with sqlite3.connect(tier_path) as conn:
                train = apply_durable_change_train(conn, train)
        except DurableChangeTrainApplyError as exc:
            _persist_train_transition(manifest_path, exc.failed_train, expected_revision=previous_revision)
            raise
        train = _persist_train_transition(manifest_path, train, expected_revision=previous_revision)
    if train.state is not DurableChangeTrainState.APPLIED:
        raise DurableChangeTrainError(f"maintenance train did not reach applied state: {train.state.value}")

    migration_result = train.apply_evidence.migration_result if train.apply_evidence is not None else None
    if legacy_result is not None and migration_result is not None:
        migration_result = MigrationResult(
            tier=tier,
            from_version=legacy_result.from_version,
            to_version=migration_result.to_version,
            applied_versions=legacy_result.applied_versions + migration_result.applied_versions,
            backup_receipt=migration_result.backup_receipt or legacy_result.backup_receipt,
        )
    previous_revision = train.revision
    train = record_durable_writer_release(
        train,
        evidence_ref=f"proof:maintenance-writer-release:{train.train_id}",
    )
    train = _persist_train_transition(manifest_path, train, expected_revision=previous_revision)

    # Keep the stable archive lease through restart and runtime proof so a
    # daemon cannot reconcile the same APPLIED manifest concurrently.
    train = _prove_and_release_persisted_train(
        archive_root,
        manifest_path,
        train,
        runtime_consumer_results=runtime_consumer_results,
    )
    release_archive_ownership()
    return DurableChangeTrainExecution(train=train, manifest_path=manifest_path, migration_result=migration_result)


def reconcile_durable_change_train_startup(archive_root: Path) -> tuple[Path, ...]:
    """Reconcile backup-authorized trains left by a crashed maintenance process."""
    from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation

    with OwnedArchiveLocation.acquire(
        ArchiveLocation.resolve(archive_root),
        owner_id=f"durable-train-recovery:{os.getpid()}",
        allow_reentrant=True,
    ):
        return _reconcile_durable_change_train_startup_locked(archive_root)


def _reconcile_durable_change_train_startup_locked(archive_root: Path) -> tuple[Path, ...]:
    """Reconcile persisted trains while the caller holds archive ownership."""
    manifest_root = archive_root / ".maintenance-state" / "durable-change-trains"
    if not manifest_root.is_dir():
        return ()
    reconciled: list[Path] = []
    for manifest_path in sorted(manifest_root.glob("*.json")):
        train = load_durable_change_train_manifest(manifest_path)
        if train.state is DurableChangeTrainState.FAILED:
            tier_path = archive_root / f"{train.tier.value}.db"
            with _open_existing_tier(tier_path) as conn:
                recovered = recover_durable_change_train(
                    conn,
                    train,
                    recovery_evidence_ref=f"proof:startup-failed-recovery:{train.train_id}",
                    writer_release_evidence_ref=f"proof:startup-writer-release:{train.train_id}",
                )
            train = _persist_train_transition(manifest_path, recovered, expected_revision=train.revision)
            reconciled.append(manifest_path)
            if train.state is DurableChangeTrainState.ADMITTED:
                continue
        if train.state is DurableChangeTrainState.BACKUP_AUTHORIZED:
            tier_path = archive_root / f"{train.tier.value}.db"
            try:
                with _open_existing_tier(tier_path) as conn:
                    recovered = reconcile_interrupted_durable_change_train(
                        conn,
                        train,
                        interruption_evidence_ref=f"proof:startup-recovery:{train.train_id}",
                        writer_release_evidence_ref=f"proof:startup-writer-release:{train.train_id}",
                    )
            except DurableChangeTrainRecoveryError as exc:
                _persist_train_transition(manifest_path, exc.failed_train, expected_revision=train.revision)
                raise
            train = _persist_train_transition(manifest_path, recovered, expected_revision=train.revision)
        if train.state is DurableChangeTrainState.RELEASED:
            with _open_existing_tier(archive_root / f"{train.tier.value}.db") as live:
                _verify_persisted_live_tier_continuity(live, train)
            reconciled.append(manifest_path)
            continue
        if train.state not in {
            DurableChangeTrainState.APPLIED,
            DurableChangeTrainState.PROVEN,
        }:
            continue
        _prove_and_release_persisted_train(archive_root, manifest_path, train)
        reconciled.append(manifest_path)
    return tuple(reconciled)


DurableChangeTrainManifest = DurableChangeTrain


def __getattr__(name: str) -> object:
    """Keep the authority import path compatible with the runner API."""
    try:
        return getattr(_migration_runner, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


__all__ = [
    "DURABLE_CHANGE_TRAIN_FORMAT",
    "DURABLE_MIGRATION_ADOPTION_FLOORS",
    "DurableChangeTrainManifest",
    "DurableMigrationSidecar",
    "DurableChangeTrainExecution",
    "durable_migration_sidecar_name",
    "validate_durable_migration_sidecars",
    "durable_change_train_policy_report",
    "durable_change_train_manifest_path",
    "durable_migration_sidecar_for_slot",
    "execute_durable_change_train",
    "reconcile_durable_change_train_startup",
    "DurableChangeTrain",
    "DurableChangeTrainState",
    "DurableChangeTrainError",
    "DurableChangeTrainApplyError",
    "DurableChangeTrainRecoveryError",
    "DurableMigrationClaim",
    "durable_migration_claim_for_sql",
    "durable_migration_claims",
    "durable_migration_collision_report",
    "find_durable_migration_collisions",
    "add_durable_change_train_rider",
    "durable_change_train_to_payload",
    "validate_durable_change_train_manifest",
    "declare_durable_change_train",
    "admit_durable_change_train",
    "reserve_durable_change_train",
    "authorize_durable_change_train_backup",
    "apply_durable_change_train",
    "recover_durable_change_train",
    "reconcile_interrupted_durable_change_train",
    "record_durable_writer_release",
    "prove_durable_change_train",
    "release_durable_change_train",
    "write_durable_change_train_manifest",
    "load_durable_change_train_manifest",
]

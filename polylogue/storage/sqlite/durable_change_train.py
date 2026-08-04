"""Durable source/user migration change-train authority."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Final

from polylogue.storage.sqlite import migration_runner as _migration_runner
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import (
    DURABLE_CHANGE_TRAIN_FORMAT,
    DurableChangeTrain,
    DurableChangeTrainApplyError,
    DurableChangeTrainError,
    DurableChangeTrainRecoveryError,
    DurableChangeTrainState,
    DurableMigrationClaim,
    _validate_riders,
    add_durable_change_train_rider,
    admit_durable_change_train,
    apply_durable_change_train,
    authorize_durable_change_train_backup,
    declare_durable_change_train,
    durable_change_train_from_payload,
    durable_change_train_to_payload,
    durable_migration_claim_for_sql,
    durable_migration_claims,
    durable_migration_collision_report,
    find_durable_migration_collisions,
    load_durable_change_train_manifest,
    prove_durable_change_train,
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
    "durable_migration_sidecar_name",
    "validate_durable_migration_sidecars",
    "durable_change_train_policy_report",
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

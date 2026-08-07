"""Durable source/user migration change-train authority."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import io
import json
import os
import re
import sqlite3
import tempfile
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Final, cast

from polylogue.storage.blob_ref_liveness import (
    BlobRefLivenessCandidate,
    BlobRefLivenessCandidateDigest,
)
from polylogue.storage.sqlite import migration_runner as _migration_runner
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import (
    DURABLE_CHANGE_TRAIN_FORMAT,
    DurableChangeTrain,
    DurableChangeTrainApplyError,
    DurableChangeTrainError,
    DurableChangeTrainRecoveryError,
    DurableChangeTrainState,
    DurableDatabaseEvidence,
    DurableFailureClassification,
    DurableFreshDDLParityProof,
    DurableMigrationClaim,
    DurableRuntimeConsumerResult,
    MigrationResult,
    _archive_identity_continuity_matches,
    _assert_durable_database_continuity,
    _canonical_json_sha256,
    _require_nonempty,
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
_SOURCE_CONTINUITY_PENDING_FORMAT = "polylogue.source-continuity-pending.v1"
_FRESH_DURABLE_BOOTSTRAP_FORMAT = "polylogue.durable-bootstrap.v1"
_FRESH_DURABLE_BOOTSTRAP_MARKER = ".bootstrap"


class DurableSourceTrainMissingError(DurableChangeTrainError):
    """Raised when an archive has no released source train to refresh."""


class DurableSourceContinuitySemanticError(DurableChangeTrainError):
    """A committed source mutation cannot satisfy immutable train evidence."""


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
    forward_version_receipt: DurableForwardVersionReceipt | None = None


@dataclass(frozen=True, slots=True)
class DurableForwardVersionReceipt:
    """Evidence that a historical released train admits a later live tier."""

    tier: ArchiveTier
    historical_train_id: str
    historical_target_version: int
    current_target_version: int
    observed_live_version: int
    historical_schema_inventory_sha256: str
    archive_identity_digest: str


@dataclass(frozen=True, slots=True)
class _DurableForwardVersionEvidence:
    """Cached live evidence reused by one no-op maintenance execution."""

    actual: DurableDatabaseEvidence
    integrity_check: tuple[str, ...]
    live_inventory: _migration_runner.DurableSchemaInventory
    canonical_inventory: _migration_runner.DurableSchemaInventory


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


def _record_fresh_durable_bootstrap(archive_root: Path) -> None:
    """Record the versions and identity of a direct, current-schema bootstrap."""
    from polylogue.storage.archive_identity import ArchiveIdentity

    archive_root = archive_root.resolve()
    marker_root = archive_root / ".maintenance-state" / "durable-change-trains"
    if (marker_root / _FRESH_DURABLE_BOOTSTRAP_MARKER).exists() or any(marker_root.glob("*.json")):
        raise DurableChangeTrainError(f"cannot record fresh durable bootstrap over existing train state: {marker_root}")
    marker_root.mkdir(parents=True, exist_ok=True)
    versions: dict[str, int] = {}
    for tier in DURABLE_MIGRATION_ADOPTION_FLOORS:
        with sqlite3.connect(archive_root / f"{tier.value}.db") as connection:
            versions[tier.value] = int(connection.execute("PRAGMA user_version").fetchone()[0])
    payload: dict[str, object] = {
        "format": _FRESH_DURABLE_BOOTSTRAP_FORMAT,
        "durable_identity_digest": _durable_identity_digest(ArchiveIdentity.resolve(archive_root)),
        "versions": versions,
    }
    payload["marker_digest"] = _bootstrap_marker_digest(payload)
    marker_path = marker_root / _FRESH_DURABLE_BOOTSTRAP_MARKER
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=marker_root,
            prefix=f".{_FRESH_DURABLE_BOOTSTRAP_MARKER}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker_path)
        temporary = None
        _migration_runner._fsync_manifest_directory(marker_root)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fresh_durable_bootstrap_versions(archive_root: Path, marker_root: Path) -> dict[ArchiveTier, int]:
    """Return direct-bootstrap versions when the marker is authentic."""
    from polylogue.storage.archive_identity import ArchiveIdentity

    archive_root = archive_root.resolve()
    marker_root = archive_root / ".maintenance-state" / "durable-change-trains"
    marker_path = marker_root / _FRESH_DURABLE_BOOTSTRAP_MARKER
    if not marker_path.is_file():
        return {}
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DurableChangeTrainError(f"invalid fresh durable bootstrap marker: {marker_path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != _FRESH_DURABLE_BOOTSTRAP_FORMAT:
        raise DurableChangeTrainError(f"fresh durable bootstrap marker format mismatch: {marker_path}")
    if payload.get("durable_identity_digest") != _durable_identity_digest(ArchiveIdentity.resolve(archive_root)):
        raise DurableChangeTrainError("fresh durable bootstrap marker durable identity mismatch")
    marker_digest = payload.get("marker_digest")
    unsigned_payload = dict(payload)
    unsigned_payload.pop("marker_digest", None)
    if not isinstance(marker_digest, str) or marker_digest != _bootstrap_marker_digest(unsigned_payload):
        raise DurableChangeTrainError("fresh durable bootstrap marker digest mismatch")
    raw_versions = payload.get("versions")
    if not isinstance(raw_versions, dict):
        raise DurableChangeTrainError(f"fresh durable bootstrap marker versions are invalid: {marker_path}")
    versions: dict[ArchiveTier, int] = {}
    for tier in DURABLE_MIGRATION_ADOPTION_FLOORS:
        raw_version = raw_versions.get(tier.value)
        if not isinstance(raw_version, int) or raw_version < 0:
            raise DurableChangeTrainError(f"fresh durable bootstrap marker version is invalid: {marker_path}")
        versions[tier] = raw_version
    return versions


def _durable_identity_digest(identity: object) -> str:
    """Digest only the durable source/user identity for bootstrap receipts."""
    from polylogue.storage.archive_identity import ArchiveIdentity

    if not isinstance(identity, ArchiveIdentity):
        raise TypeError("durable identity digest requires an ArchiveIdentity")
    payload = {
        "configured_root": str(identity.configured_root.absolute()),
        "durable_id": identity.durable_id,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _bootstrap_marker_digest(payload: dict[str, object]) -> str:
    """Authenticate bootstrap identity and recorded durable versions together."""
    return _canonical_json_sha256(payload)


def _adopt_pre_marker_durable_bootstrap(archive_root: Path) -> None:
    """Authenticate a current-schema archive created before bootstrap receipts."""
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER

    archive_root = archive_root.resolve()
    manifest_root = archive_root / ".maintenance-state" / "durable-change-trains"
    if (manifest_root / _FRESH_DURABLE_BOOTSTRAP_MARKER).is_file():
        return
    if any(manifest_root.glob("*.json")):
        return
    for tier in DURABLE_MIGRATION_ADOPTION_FLOORS:
        tier_path = archive_root / f"{tier.value}.db"
        if not tier_path.is_file():
            continue
        with _open_existing_tier(tier_path) as connection:
            current_version = int(connection.execute("PRAGMA user_version").fetchone()[0] or 0)
            expected_version = ARCHIVE_VERSION_BY_TIER[tier]
            if current_version != expected_version:
                raise DurableChangeTrainError(
                    f"pre-marker {tier.value} durable tier is v{current_version}, expected current v{expected_version}"
                )
            actual_inventory = _migration_runner.capture_durable_schema_inventory(connection)
            expected_inventory = _canonical_schema_inventory(tier, expected_version)
            if actual_inventory.sha256 != expected_inventory.sha256:
                raise DurableChangeTrainError(
                    f"pre-marker {tier.value} durable tier schema does not match current canonical DDL"
                )
    _record_fresh_durable_bootstrap(archive_root)


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


def write_source_continuity_pending_intent(
    archive_root: Path,
    *,
    mutation_receipt: Path,
    backup_manifest: Path,
    pre_mutation_evidence: DurableDatabaseEvidence,
    operation_id: str,
    evidence_ref: str,
) -> Path:
    """Persist the recovery input before a source mutation can commit."""
    mutation_receipt = mutation_receipt.resolve()
    backup_manifest = backup_manifest.resolve()
    pending_root = archive_root / ".maintenance-state" / "source-continuity-pending"
    pending_root_existed = pending_root.is_dir()
    pending_root.mkdir(parents=True, exist_ok=True)
    if not pending_root_existed:
        _migration_runner._fsync_manifest_directory(pending_root.parent)
    payload: dict[str, object] = {
        "format": _SOURCE_CONTINUITY_PENDING_FORMAT,
        "mutation_receipt": str(mutation_receipt),
        "backup_manifest": str(backup_manifest),
        "operation_id": operation_id,
        "evidence_ref": evidence_ref,
        "source_before": _migration_runner._manifest_json_value(pre_mutation_evidence),
    }
    pending_digest = _canonical_json_sha256(payload)
    path = pending_root / f"{pending_digest}.json"
    encoded = (json.dumps({**payload, "pending_sha256": pending_digest}, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DurableChangeTrainError(f"source continuity pending intent is unreadable: {path}") from exc
        if existing != {**payload, "pending_sha256": pending_digest}:
            raise DurableChangeTrainError("source continuity pending intent collision")
        return path
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=pending_root, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _migration_runner._fsync_manifest_directory(pending_root)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _load_source_continuity_pending_intent(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DurableChangeTrainError(f"source continuity pending intent is unreadable: {path}") from exc
    if not isinstance(raw, dict):
        raise DurableChangeTrainError(f"source continuity pending intent is not an object: {path}")
    pending_digest = raw.pop("pending_sha256", None)
    if not isinstance(pending_digest, str) or pending_digest != _canonical_json_sha256(raw):
        raise DurableChangeTrainError(f"source continuity pending intent checksum mismatch: {path}")
    if raw.get("format") != _SOURCE_CONTINUITY_PENDING_FORMAT:
        raise DurableChangeTrainError(f"unsupported source continuity pending intent: {path}")
    return cast(dict[str, object], raw)


def _replace_source_continuity_pending_intent(path: Path, payload: dict[str, object]) -> None:
    encoded = (
        json.dumps({**payload, "pending_sha256": _canonical_json_sha256(payload)}, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _migration_runner._fsync_manifest_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def mark_source_continuity_pending_intent_terminal(path: Path, *, error: DurableSourceContinuitySemanticError) -> None:
    """Persist a semantic refresh rejection so startup does not retry it forever."""
    payload = _load_source_continuity_pending_intent(path)
    terminal = {"kind": "continuity_refresh_rejected", "error": str(error)}
    existing = payload.get("terminal_outcome")
    if existing == terminal:
        return
    if existing is not None:
        raise DurableChangeTrainError(f"source continuity pending intent has an unknown terminal outcome: {path}")
    _replace_source_continuity_pending_intent(path, {**payload, "terminal_outcome": terminal})


def clear_source_continuity_pending_intent(path: Path) -> None:
    """Remove a consumed pending intent only after manifest refresh succeeds."""
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _migration_runner._fsync_manifest_directory(path.parent)


def assert_source_continuity_apply_allowed(archive_root: Path) -> None:
    """Reject a new source mutation that could invalidate continuity recovery."""
    archive_root = archive_root.resolve()
    pending_root = archive_root / ".maintenance-state" / "source-continuity-pending"
    pending_intents = tuple(sorted(pending_root.glob("*.json"))) if pending_root.is_dir() else ()
    if pending_intents:
        raise DurableChangeTrainError("source liveness apply is blocked while source continuity recovery is pending")

    source_path = archive_root / "source.db"
    with sqlite3.connect(f"file:{source_path}?mode=ro", uri=True) as connection:
        current_version = int(connection.execute("PRAGMA user_version").fetchone()[0] or 0)
    manifest_root = archive_root / ".maintenance-state" / "durable-change-trains"
    if not manifest_root.is_dir():
        return
    unreleased: list[Path] = []
    released: list[DurableChangeTrain] = []
    for candidate in sorted(manifest_root.glob("source-*.json")):
        train = load_durable_change_train_manifest(candidate)
        rollback_failed_train = (
            train.state is DurableChangeTrainState.FAILED
            and train.failure is not None
            and train.failure.classification is DurableFailureClassification.ROLLED_BACK_TO_CURRENT
            and train.current_version == current_version
        )
        if (
            train.target_version != current_version
            and not (
                train.reservation is not None and train.reservation.active and train.current_version == current_version
            )
            and not rollback_failed_train
        ):
            continue
        if train.state is DurableChangeTrainState.RELEASED:
            released.append(train)
        else:
            unreleased.append(candidate)
    if unreleased:
        raise DurableChangeTrainError(
            "source liveness apply is blocked by an unreleased source train for the live schema"
        )
    if len(released) > 1:
        raise DurableChangeTrainError(
            "source liveness apply requires exactly one released source train for the live schema"
        )
    if released:
        with sqlite3.connect(f"file:{source_path}?mode=ro", uri=True) as connection:
            _verify_released_train_live_tier(archive_root, connection, released[0])


def _recover_pending_source_continuity_intents(archive_root: Path) -> None:
    """Finish committed source mutations whose manifest refresh was interrupted."""
    pending_root = archive_root / ".maintenance-state" / "source-continuity-pending"
    if not pending_root.is_dir():
        return
    for path in sorted(pending_root.glob("*.json")):
        raw = _load_source_continuity_pending_intent(path)
        terminal = raw.get("terminal_outcome")
        if terminal is not None:
            if not (
                isinstance(terminal, dict)
                and terminal.get("kind") == "continuity_refresh_rejected"
                and isinstance(terminal.get("error"), str)
            ):
                raise DurableChangeTrainError(
                    f"source continuity pending intent has an invalid terminal outcome: {path}"
                )
            clear_source_continuity_pending_intent(path)
            continue
        try:
            evidence_raw = raw["source_before"]
            if not isinstance(evidence_raw, dict):
                raise TypeError("source_before is not an object")
            pre_mutation_evidence = _migration_runner._decode_manifest_value(
                DurableDatabaseEvidence,
                evidence_raw,
                label=f"{path}.source_before",
            )
            if not isinstance(pre_mutation_evidence, DurableDatabaseEvidence):
                raise TypeError("source_before decoded to the wrong type")
            receipt = Path(str(raw["mutation_receipt"]))
            backup = Path(str(raw["backup_manifest"]))
            operation_id = str(raw["operation_id"])
            evidence_ref = str(raw["evidence_ref"])
        except (DurableChangeTrainError, KeyError, TypeError, ValueError) as exc:
            raise DurableChangeTrainError(f"source continuity pending intent is malformed: {path}") from exc
        receipt_phase = _liveness_receipt_phase(receipt)
        if receipt_phase == "recovered_rolled_back":
            clear_source_continuity_pending_intent(path)
            continue
        if receipt_phase in {"prepared", "batch_committed"}:
            from polylogue.maintenance.blob_ref_liveness_reconciliation import _recover_prepared_receipt

            outcome = _recover_prepared_receipt(archive_root / "source.db", receipt)
            if outcome == "recovered_rolled_back":
                clear_source_continuity_pending_intent(path)
                continue
            if outcome == "recovered_partial":
                raise DurableChangeTrainError(f"source continuity pending intent has a partial source mutation: {path}")
            receipt_phase = outcome
        if receipt_phase == "postcondition_failed":
            from polylogue.maintenance.blob_ref_liveness_reconciliation import _recover_prepared_receipt
            from polylogue.storage.blob_ref_liveness import classify_blob_ref_liveness

            def validate_recovered_postcondition(pending_path: Path = path) -> None:
                with sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True) as connection:
                    classification = classify_blob_ref_liveness(connection)
                    if not classification.safe_to_apply or classification.orphaned_count != 0:
                        raise DurableChangeTrainError(
                            f"source continuity pending intent postcondition remains unsafe: {pending_path}"
                        )

            outcome = _recover_prepared_receipt(
                archive_root / "source.db",
                receipt,
                allow_postcondition_failed=True,
                postcondition_check=validate_recovered_postcondition,
            )
            if outcome == "recovered_rolled_back":
                clear_source_continuity_pending_intent(path)
                continue
            if outcome == "recovered_partial":
                raise DurableChangeTrainError(f"source continuity pending intent has a partial source mutation: {path}")
            receipt_phase = outcome
        if receipt_phase not in {"committed", "recovered_committed"}:
            raise DurableChangeTrainError(f"source continuity pending intent has no committed receipt: {path}")
        try:
            refresh_released_source_train_continuity(
                archive_root,
                mutation_receipt=receipt,
                backup_manifest=backup,
                pre_mutation_evidence=pre_mutation_evidence,
                operation_id=operation_id,
                evidence_ref=evidence_ref,
            )
        except DurableSourceTrainMissingError:
            clear_source_continuity_pending_intent(path)
        except DurableSourceContinuitySemanticError as exc:
            mark_source_continuity_pending_intent_terminal(path, error=exc)
        else:
            clear_source_continuity_pending_intent(path)


def _liveness_receipt_phase(receipt_path: Path) -> str:
    """Read the last liveness phase before deciding how a pending intent recovers."""
    try:
        last_record: object | None = None
        with receipt_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last_record = json.loads(line)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DurableChangeTrainError(f"source continuity pending receipt is unreadable: {receipt_path}") from exc
    if not isinstance(last_record, dict):
        raise DurableChangeTrainError(f"source continuity pending receipt is incomplete: {receipt_path}")
    phase = last_record.get("phase")
    if not isinstance(phase, str):
        raise DurableChangeTrainError(f"source continuity pending receipt has no phase: {receipt_path}")
    return phase


def _validate_liveness_receipt_bytes(
    receipt_bytes: bytes,
    *,
    source_path: Path,
    backup_manifest: Path,
    operation_id: str,
) -> dict[str, object]:
    """Validate the exact candidate stream and terminal footer of a liveness receipt."""
    header: dict[str, object] | None = None
    footer: dict[str, object] | None = None
    candidate_digest = BlobRefLivenessCandidateDigest()
    candidate_count = 0
    try:
        for raw_line in io.BytesIO(receipt_bytes):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise DurableChangeTrainError("source mutation receipt contains a non-object record")
            if header is None:
                header = cast(dict[str, object], record)
                continue
            if footer is not None:
                row = footer
                if row.get("kind") == "blob_ref_liveness_reconciliation":
                    previous_phase = row.get("phase")
                    current_phase = (
                        record.get("phase") if record.get("kind") == "blob_ref_liveness_reconciliation" else None
                    )
                    if not (
                        previous_phase == "batch_committed"
                        or previous_phase == "postcondition_failed"
                        and current_phase == "recovered_committed"
                    ):
                        raise DurableChangeTrainError(
                            "source mutation receipt contains an unexpected intermediate footer"
                        )
                elif row.get("kind") == "candidate":
                    try:
                        blob_hash = str(row["blob_hash"])
                        bytes.fromhex(blob_hash)
                        size_bytes = row["size_bytes"]
                        acquired_at_ms = row["acquired_at_ms"]
                        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool):
                            raise TypeError("candidate size_bytes is not an integer")
                        if not isinstance(acquired_at_ms, int) or isinstance(acquired_at_ms, bool):
                            raise TypeError("candidate acquired_at_ms is not an integer")
                        candidate_digest.update(
                            BlobRefLivenessCandidate(
                                blob_hash=blob_hash,
                                ref_type=str(row["ref_type"]),
                                ref_id=str(row["ref_id"]),
                                source_path=str(row["source_path"]) if row.get("source_path") is not None else None,
                                size_bytes=size_bytes,
                                acquired_at_ms=acquired_at_ms,
                                referent_table=str(row["referent_table"]),
                                referent_column=str(row["referent_column"]),
                            )
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        raise DurableChangeTrainError("source mutation receipt contains an invalid candidate") from exc
                    candidate_count += 1
                else:
                    raise DurableChangeTrainError("source mutation receipt contains an unexpected record")
            footer = cast(dict[str, object], record)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DurableChangeTrainError("source mutation receipt is not valid JSONL") from exc
    if header is None or footer is None:
        raise DurableChangeTrainError("source mutation receipt is incomplete")
    if (
        header.get("kind") != "blob_ref_liveness_reconciliation"
        or header.get("phase") != "prepared"
        or header.get("source_db") != str(source_path)
        or header.get("backup_manifest") != str(backup_manifest)
        or header.get("candidate_digest") != operation_id
        or footer.get("kind") != "blob_ref_liveness_reconciliation"
        or footer.get("phase") not in {"committed", "recovered_committed"}
        or footer.get("deleted_count") != header.get("candidate_count")
        or (footer.get("phase") == "committed" and footer.get("post_orphaned_count") != 0)
    ):
        raise DurableChangeTrainError("source mutation receipt does not bind the named liveness operation")

    if (
        header.get("candidate_count") != candidate_count
        or header.get("candidate_digest") != candidate_digest.hexdigest()
    ):
        raise DurableChangeTrainError("source mutation receipt candidate digest or count mismatch")
    return header


def _validate_source_continuity_refresh_receipt(
    archive_root: Path,
    train: DurableChangeTrain,
) -> None:
    """Require the latest source continuity evidence to retain its receipt."""
    if train.source_continuity_evidence is None:
        return
    expected_after = _migration_runner._manifest_json_value(train.source_continuity_evidence)
    refresh_root = archive_root / ".maintenance-state" / "source-continuity-refreshes"
    refresh_refs = [
        ref.removeprefix("proof:source-continuity-refresh:")
        for ref in train.proof_refs
        if ref.startswith("proof:source-continuity-refresh:")
    ]
    if not refresh_refs:
        raise DurableChangeTrainError("source continuity evidence has no retained refresh receipt")
    matches = 0
    for digest in refresh_refs:
        receipt_path = refresh_root / f"{digest}.json"
        payload = _read_source_continuity_refresh_receipt(receipt_path, digest=digest, train=train)
        if payload.get("source_after") == expected_after:
            matches += 1
    if matches != 1:
        raise DurableChangeTrainError(
            "source continuity evidence does not identify exactly one matching refresh receipt"
        )


def _read_source_continuity_refresh_receipt(
    receipt_path: Path,
    *,
    digest: str,
    train: DurableChangeTrain,
) -> dict[str, object]:
    """Load one train-retained refresh artifact and authenticate its identity."""
    try:
        raw = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DurableChangeTrainError(f"source continuity refresh receipt is unreadable: {receipt_path}") from exc
    if not isinstance(raw, dict):
        raise DurableChangeTrainError(f"source continuity refresh receipt is not an object: {receipt_path}")
    payload = cast(dict[str, object], raw)
    refresh_sha256 = payload.pop("refresh_sha256", None)
    if refresh_sha256 != digest or _canonical_json_sha256(payload) != digest:
        raise DurableChangeTrainError(f"source continuity refresh receipt checksum mismatch: {receipt_path}")
    if payload.get("format") != "polylogue.source-continuity-refresh.v1":
        raise DurableChangeTrainError(f"source continuity refresh receipt format mismatch: {receipt_path}")
    if payload.get("train_id") != train.train_id:
        raise DurableChangeTrainError(f"source continuity refresh receipt train mismatch: {receipt_path}")
    return payload


def refresh_released_source_train_continuity(
    archive_root: Path,
    *,
    mutation_receipt: Path,
    backup_manifest: Path,
    pre_mutation_evidence: DurableDatabaseEvidence,
    operation_id: str,
    evidence_ref: str,
) -> Path:
    """Refresh released source-train continuity while retaining archive ownership."""
    from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation

    with OwnedArchiveLocation.acquire(
        ArchiveLocation.resolve(archive_root),
        owner_id=f"source-continuity-refresh:{os.getpid()}",
        allow_reentrant=True,
    ):
        return _refresh_released_source_train_continuity_locked(
            archive_root,
            mutation_receipt=mutation_receipt,
            backup_manifest=backup_manifest,
            pre_mutation_evidence=pre_mutation_evidence,
            operation_id=operation_id,
            evidence_ref=evidence_ref,
        )


def _refresh_released_source_train_continuity_locked(
    archive_root: Path,
    *,
    mutation_receipt: Path,
    backup_manifest: Path,
    pre_mutation_evidence: DurableDatabaseEvidence,
    operation_id: str,
    evidence_ref: str,
) -> Path:
    """Record an authorized source mutation without weakening train checks.

    Source maintenance may change rows after a schema train is released. The
    caller must first validate the named operation and backup against the
    pre-mutation live tier. This helper then binds those exact receipt and
    backup bytes to a separate current-evidence record. The original migration
    evidence remains immutable in ``apply_evidence``; only the
    legacy-authority-digest compatibility path may rewrite that historical
    identity field while preserving the migration evidence.
    """
    from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation, OwnedArchiveLocation

    archive_root = archive_root.resolve()
    mutation_receipt = mutation_receipt.resolve()
    backup_manifest = backup_manifest.resolve()

    _require_nonempty(operation_id, label="source mutation operation id")
    _require_nonempty(evidence_ref, label="source continuity evidence ref")
    if not mutation_receipt.is_file() or mutation_receipt.is_symlink():
        raise DurableChangeTrainError("source mutation receipt is not a real file")
    if not backup_manifest.is_file() or backup_manifest.is_symlink():
        raise DurableChangeTrainError("source mutation backup manifest is not a real file")

    source_path = archive_root / "source.db"
    try:
        receipt_bytes = mutation_receipt.read_bytes()
    except OSError as exc:
        raise DurableChangeTrainError("source mutation receipt is not readable") from exc
    header = _validate_liveness_receipt_bytes(
        receipt_bytes,
        source_path=source_path,
        backup_manifest=backup_manifest,
        operation_id=operation_id,
    )

    mutation_digest = hashlib.sha256(receipt_bytes).hexdigest()
    backup_digest = hashlib.sha256(backup_manifest.read_bytes()).hexdigest()
    receipt_backup_digest = header.get("backup_manifest_sha256")
    if not isinstance(receipt_backup_digest, str) or receipt_backup_digest != backup_digest:
        raise DurableChangeTrainError("source mutation receipt backup manifest digest mismatch")
    with OwnedArchiveLocation.acquire(
        ArchiveLocation.resolve(archive_root),
        owner_id=f"source-continuity-refresh:{os.getpid()}",
        allow_reentrant=True,
    ):
        with sqlite3.connect(f"file:{source_path}?mode=ro", uri=True) as connection:
            current = capture_durable_database_evidence(connection, ArchiveTier.SOURCE)

        manifest_candidates = sorted(
            (archive_root / ".maintenance-state" / "durable-change-trains").glob("source-*.json")
        )
        if not manifest_candidates:
            raise DurableSourceTrainMissingError("source continuity refresh found no released source train")
        matching: list[tuple[Path, DurableChangeTrain]] = []
        for candidate in manifest_candidates:
            candidate_train = load_durable_change_train_manifest(candidate)
            if (
                candidate_train.state is DurableChangeTrainState.RELEASED
                and candidate_train.tier is ArchiveTier.SOURCE
                and candidate_train.target_version == current.user_version
            ):
                matching.append((candidate, candidate_train))
        if not matching:
            raise DurableSourceTrainMissingError("source continuity refresh found no released source train")
        if len(matching) != 1:
            raise DurableChangeTrainError(
                "source continuity refresh requires exactly one released source train for the live schema"
            )
        manifest_path, train = matching[0]
        if train.state is not DurableChangeTrainState.RELEASED:
            raise DurableChangeTrainError("source continuity refresh requires a released source train")
        if train.tier is not ArchiveTier.SOURCE:
            raise DurableChangeTrainError("source continuity refresh selected a non-source train")
        if train.apply_evidence is None:
            raise DurableChangeTrainError("source continuity refresh requires apply evidence")
        refresh_root = archive_root / ".maintenance-state" / "source-continuity-refreshes"
        serialized_before = _migration_runner._manifest_json_value(pre_mutation_evidence)
        serialized_current = _migration_runner._manifest_json_value(current)
        retained_current = train.source_continuity_evidence
        current_matches_retained = False
        if retained_current is not None:
            try:
                _migration_runner._assert_durable_database_continuity(
                    current,
                    retained_current,
                    label="source continuity retained refresh",
                    archive_root=archive_root,
                )
            except DurableChangeTrainError:
                pass
            else:
                current_matches_retained = True
        serialized_retained_current = (
            _migration_runner._manifest_json_value(retained_current) if retained_current is not None else None
        )
        retained_refreshes: list[tuple[Path, dict[str, object]]] = []
        retained_refs = {
            ref.removeprefix("proof:source-continuity-refresh:")
            for ref in train.proof_refs
            if ref.startswith("proof:source-continuity-refresh:")
        }
        for existing_path in sorted(refresh_root.glob("*.json")) if refresh_root.is_dir() else ():
            digest = existing_path.stem
            if digest not in retained_refs:
                continue
            existing = _read_source_continuity_refresh_receipt(existing_path, digest=digest, train=train)
            if existing.get("mutation_receipt_sha256") == mutation_digest:
                retained_refreshes.append((existing_path, existing))
        for existing_path, existing in retained_refreshes:
            # A crash after manifest persistence can leave its pending intent
            # behind. The exact retained artifact, including its before/after
            # evidence, authenticates this idempotent completion.
            if (
                current_matches_retained
                and existing.get("source_before") == serialized_before
                and existing.get("source_after") == serialized_retained_current
            ):
                _validate_source_continuity_refresh_receipt(archive_root, train)
                return existing_path
        if pre_mutation_evidence.user_version != train.target_version:
            raise DurableSourceContinuitySemanticError(
                "source continuity refresh pre-state has the wrong schema version"
            )
        if current.user_version != train.target_version:
            raise DurableSourceContinuitySemanticError("source continuity refresh changed the schema version")
        baseline = train.source_continuity_evidence or train.apply_evidence.post
        try:
            _migration_runner._assert_durable_database_continuity(
                pre_mutation_evidence,
                baseline,
                label="source continuity pre-mutation",
                archive_root=archive_root,
            )
        except DurableChangeTrainError as exc:
            raise DurableSourceContinuitySemanticError(
                "source continuity refresh pre-state contains unreceipted content drift"
            ) from exc
        if not _archive_identity_continuity_matches(
            pre_mutation_evidence.archive_identity_digest,
            train.apply_evidence.post.archive_identity_digest,
            archive_root,
            ArchiveTier.SOURCE,
        ):
            raise DurableSourceContinuitySemanticError(
                "source continuity refresh pre-state has the wrong archive identity"
            )
        if not _archive_identity_continuity_matches(
            current.archive_identity_digest,
            train.apply_evidence.post.archive_identity_digest,
            archive_root,
            ArchiveTier.SOURCE,
        ):
            raise DurableSourceContinuitySemanticError("source continuity refresh changed archive identity")
        if pre_mutation_evidence.quick_check != ("ok",) or current.quick_check != ("ok",):
            raise DurableSourceContinuitySemanticError(
                "source continuity refresh requires successful quick_check evidence"
            )

        payload = {
            "format": "polylogue.source-continuity-refresh.v1",
            "operation_id": operation_id,
            "evidence_ref": evidence_ref,
            "backup_manifest": str(backup_manifest),
            "backup_manifest_sha256": backup_digest,
            "mutation_receipt": str(mutation_receipt),
            "mutation_receipt_sha256": mutation_digest,
            "train_id": train.train_id,
            "source_before": serialized_before,
            "source_after": serialized_current,
            "refreshed_at_ms": current.observed_at_ms,
        }
        refresh_digest = _canonical_json_sha256(payload)
        refresh_root_existed = refresh_root.is_dir()
        refresh_root.mkdir(parents=True, exist_ok=True)
        if not refresh_root_existed:
            _migration_runner._fsync_manifest_directory(refresh_root.parent)
        refresh_path = refresh_root / f"{refresh_digest}.json"
        if refresh_path.exists():
            try:
                existing = json.loads(refresh_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DurableChangeTrainError(
                    f"source continuity refresh receipt is unreadable: {refresh_path}"
                ) from exc
            if existing != {**payload, "refresh_sha256": refresh_digest}:
                raise DurableChangeTrainError("source continuity refresh receipt collision")
        else:
            encoded = (
                json.dumps({**payload, "refresh_sha256": refresh_digest}, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=refresh_root,
                    prefix=f".{refresh_path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, refresh_path)
                temporary = None
                _migration_runner._fsync_manifest_directory(refresh_root)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        refresh_ref = f"proof:source-continuity-refresh:{refresh_digest}"
        references = _migration_runner._append_proof_refs(train.proof_refs, evidence_ref, refresh_ref)
        if train.proof is None:
            raise DurableChangeTrainError("source continuity refresh requires train proof")
        retained_apply_evidence = train.apply_evidence
        legacy_archive_identity_digest = ArchiveIdentity.resolve(archive_root).authority_identity_digest
        if (
            retained_apply_evidence.post.archive_identity_digest == legacy_archive_identity_digest
            and current.archive_identity_digest != retained_apply_evidence.post.archive_identity_digest
        ):
            retained_apply_evidence = replace(
                retained_apply_evidence,
                post=replace(
                    retained_apply_evidence.post,
                    archive_identity_digest=current.archive_identity_digest,
                ),
            )
        updated = replace(
            train,
            revision=train.revision + 1,
            apply_evidence=retained_apply_evidence,
            source_continuity_evidence=current,
            proof_refs=references,
        )
        write_durable_change_train_manifest(manifest_path, updated, expected_revision=train.revision)
    return refresh_path


def _fresh_ddl_parity_for_train(
    train: DurableChangeTrain,
    *,
    migrated_connection: sqlite3.Connection | None = None,
) -> DurableFreshDDLParityProof:
    """Compare a live result, or two canonical creates, against bootstrap DDL."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier

    def prepare_target_schema(connection: sqlite3.Connection) -> None:
        _migration_runner._prepare_fresh_connection_for_target(connection, train.tier, train.target_version)

    if migrated_connection is None:
        with sqlite3.connect(":memory:") as migrated, sqlite3.connect(":memory:") as fresh:
            initialize_archive_tier(migrated, train.tier)
            initialize_archive_tier(fresh, train.tier)
            prepare_target_schema(migrated)
            prepare_target_schema(fresh)
            return prove_durable_fresh_ddl_parity(
                train.tier,
                train.target_version,
                migrated_connection=migrated,
                fresh_connection=fresh,
                evidence_ref=f"proof:canonical-bootstrap:{train.tier.value}:v{train.target_version}",
            )
    with sqlite3.connect(":memory:") as fresh:
        initialize_archive_tier(fresh, train.tier)
        prepare_target_schema(fresh)
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
            detail = f"resolved {reference}"
            try:
                if reference.endswith(":initialize_archive_database"):
                    tier_path = archive_root / f"{train.tier.value}.db"
                    with _open_existing_tier(tier_path) as probe:
                        live_version = int(probe.execute("PRAGMA user_version").fetchone()[0] or 0)
                    runtime_target = cast(dict[ArchiveTier, int], vars(_migration_runner)["ARCHIVE_VERSION_BY_TIER"])[
                        train.tier
                    ]
                    if live_version == runtime_target:
                        value(tier_path, train.tier, allow_create=False)
                    else:
                        value(tier_path, train.tier, allow_create=False, expected_version=train.target_version)
                elif reference.endswith(":initialize_archive_tier"):
                    with sqlite3.connect(":memory:") as probe:
                        value(probe, train.tier)
                elif reference.endswith(":write_source_hook_event"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_source_hook_event_writer(cast(Callable[..., object], value))
                elif reference.endswith(":_stage_locked_hook_snapshot"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_locked_hook_snapshot(cast(Callable[..., object], value), train.target_version)
                elif reference.endswith(":_create_match_stage"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_hook_match_stage(cast(Callable[..., object], value), train.target_version)
                elif reference.endswith(":read_raw_failure_lifecycle"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_raw_failure_lifecycle(cast(Callable[..., object], value), archive_root)
                elif reference.endswith(":apply_raw_failure_dispositions"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_raw_failure_disposition_apply(cast(Callable[..., object], value), archive_root)
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
                    detail=detail,
                )
            )
    return tuple(results)


def _probe_source_hook_event_writer(writer: Callable[..., object]) -> str:
    """Exercise the source hook writer against an isolated fresh source tier."""
    from polylogue.core.enums import Origin
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.source_write import (
        ArchiveHookEvent,
        deterministic_blob_hash,
    )

    source_path = "/durable-change-train/source-v27-probe.jsonl"
    payload = b'{"event":"PostToolUse","probe":"source-v27"}'
    hook_event = ArchiveHookEvent(
        hook_event_id="durable-change-train-source-v27-hook",
        origin=Origin.CODEX_SESSION,
        source_path=source_path,
        event_type="PostToolUse",
        payload={"event": "PostToolUse", "probe": "source-v27"},
        observed_at_ms=1_780_000_000_000,
        native_id="durable-change-train-source-v27-native",
        session_native_id="durable-change-train-source-v27-session",
    )
    expected_blob_hash = deterministic_blob_hash(payload)
    with sqlite3.connect(":memory:") as probe:
        initialize_archive_tier(probe, ArchiveTier.SOURCE)
        returned_raw_id = writer(
            probe,
            origin=hook_event.origin,
            source_path=source_path,
            payload=payload,
            acquired_at_ms=hook_event.observed_at_ms,
            raw_id="durable-change-train-source-v27-raw",
            hook_event=hook_event,
        )
        hook_row = probe.execute(
            """
            SELECT origin, native_id, session_native_id, source_path, event_type,
                   payload_json, observed_at_ms, blob_hash
            FROM raw_hook_events
            WHERE hook_event_id = ?
            """,
            (hook_event.hook_event_id,),
        ).fetchone()
        blob_ref_row = probe.execute(
            "SELECT blob_hash, ref_type, ref_id, source_path, size_bytes, acquired_at_ms FROM blob_refs"
        ).fetchone()
        raw_session_count = probe.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()

    expected_hook_row = (
        Origin.CODEX_SESSION.value,
        hook_event.native_id,
        hook_event.session_native_id,
        source_path,
        hook_event.event_type,
        '{"event":"PostToolUse","probe":"source-v27"}',
        hook_event.observed_at_ms,
        expected_blob_hash,
    )
    expected_blob_ref_row = (
        expected_blob_hash,
        "hook_payload",
        hook_event.hook_event_id,
        source_path,
        len(payload),
        hook_event.observed_at_ms,
    )
    if returned_raw_id != "durable-change-train-source-v27-raw":
        raise DurableChangeTrainError("source hook writer probe returned the wrong raw identity")
    if hook_row != expected_hook_row or blob_ref_row != expected_blob_ref_row or raw_session_count != (0,):
        raise DurableChangeTrainError("source hook writer probe did not persist the expected hook payload contract")
    return "wrote and read back a hook payload in a fresh source tier"


def _runtime_probe_source_connection(target_version: int) -> sqlite3.Connection:
    """Create a source-tier probe projected to the train's schema slot."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier

    connection = sqlite3.connect(":memory:")
    initialize_archive_tier(connection, ArchiveTier.SOURCE)
    _migration_runner._prepare_fresh_connection_for_target(connection, ArchiveTier.SOURCE, target_version)
    return connection


def _seed_hook_reconciliation_probe(connection: sqlite3.Connection) -> tuple[str, bytes, str]:
    """Install one deterministic orphaned raw payload and its hook evidence."""
    from polylogue.core.enums import Origin
    from polylogue.storage.sqlite.archive_tiers.source_write import (
        deterministic_blob_hash,
        deterministic_raw_session_id,
    )

    source_path = "/durable-change-train/hook-reconciliation-probe.jsonl"
    payload = b'{"event":"PostToolUse","probe":"durable-change-train"}'
    blob_hash = deterministic_blob_hash(payload)
    native_id = "durable-change-train-hook-native"
    ref_id = deterministic_raw_session_id(Origin.CODEX_SESSION, source_path, 0, blob_hash, native_id)
    connection.execute(
        """
        INSERT INTO raw_hook_events (
            hook_event_id, origin, native_id, session_native_id, source_path,
            event_type, payload_json, observed_at_ms, blob_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "durable-change-train-hook-event",
            Origin.CODEX_SESSION.value,
            native_id,
            "durable-change-train-hook-session",
            source_path,
            "PostToolUse",
            payload.decode("utf-8"),
            1_780_000_000_000,
            blob_hash,
        ),
    )
    connection.execute(
        """
        INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
        VALUES (?, ?, 'raw_payload', ?, ?, ?)
        """,
        (blob_hash, ref_id, source_path, len(payload), 1_780_000_000_000),
    )
    connection.execute(
        """
        CREATE TEMP TABLE durable_change_train_probe_candidates (
            blob_hash BLOB NOT NULL,
            ref_type TEXT NOT NULL,
            ref_id TEXT NOT NULL,
            source_path TEXT,
            size_bytes INTEGER NOT NULL,
            acquired_at_ms INTEGER NOT NULL
        ) STRICT
        """
    )
    connection.execute(
        """
        INSERT INTO durable_change_train_probe_candidates
        SELECT blob_hash, ref_type, ref_id, source_path, size_bytes, acquired_at_ms
        FROM blob_refs
        WHERE ref_type = 'raw_payload'
        """
    )
    connection.commit()
    return "durable_change_train_probe_candidates", payload, ref_id


def _probe_locked_hook_snapshot(snapshot: Callable[..., object], target_version: int) -> str:
    """Exercise liveness snapshotting against a real source-tier fixture."""
    with _runtime_probe_source_connection(target_version) as connection:
        candidate_table, _payload, _ref_id = _seed_hook_reconciliation_probe(connection)
        returned_table = snapshot(connection, candidate_table)
        locked_count = int(connection.execute("SELECT COUNT(*) FROM temp.blob_ref_liveness_locked_hooks").fetchone()[0])
        identity_count = int(
            connection.execute("SELECT COUNT(*) FROM temp.blob_ref_liveness_locked_identity_matches").fetchone()[0]
        )
    if returned_table != "blob_ref_liveness_locked_hooks" or locked_count != 1 or identity_count != 1:
        raise DurableChangeTrainError(
            "liveness hook snapshot probe did not preserve the expected candidate identity evidence"
        )
    return "staged one hook candidate with one identity match"


def _probe_hook_match_stage(match_stage: Callable[..., object], target_version: int) -> str:
    """Exercise hook match staging against a real orphan/reference fixture."""
    with _runtime_probe_source_connection(target_version) as connection:
        _candidate_table, payload, ref_id = _seed_hook_reconciliation_probe(connection)
        result = match_stage(connection)
    if result != (1, 1, len(payload), 0):
        raise DurableChangeTrainError(f"hook match-stage probe produced unexpected counts for {ref_id}: {result!r}")
    return "staged one orphan with one unambiguous hook match"


def _probe_raw_failure_lifecycle(reader: Callable[..., object], archive_root: Path) -> str:
    """Exercise the source-tier failure lifecycle reader against live bytes."""
    snapshot = reader(archive_root / "source.db", sample_limit=1)
    if not getattr(snapshot, "available", False):
        raise DurableChangeTrainError("raw failure lifecycle probe could not read source.db")
    return f"read raw failure lifecycle state={getattr(snapshot, 'state', 'unknown')}"


def _probe_raw_failure_disposition_apply(actuator: Callable[..., object], archive_root: Path) -> str:
    """Exercise the disposition actuator's read-only validation route."""
    del archive_root
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier

    with tempfile.TemporaryDirectory(prefix="polylogue-durable-train-disposition-") as directory:
        root = Path(directory)
        source_path = root / "source.db"
        with sqlite3.connect(source_path) as connection:
            initialize_archive_tier(connection, ArchiveTier.SOURCE)
            connection.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, source_path, source_index, blob_hash, blob_size,
                    acquired_at_ms, parse_error
                ) VALUES (?, ?, ?, 0, ?, 0, ?, ?)
                """,
                (
                    "durable-change-train-disposition-raw",
                    "claude-code-session",
                    "/durable-change-train/disposition-probe.jsonl",
                    b"\0" * 32,
                    1_780_000_000_000,
                    "durable change train probe failure",
                ),
            )
            connection.execute(
                """
                INSERT INTO raw_artifacts (
                    artifact_id, raw_id, origin, source_path, source_index,
                    artifact_kind, support_status, classification_reason,
                    first_observed_at_ms, last_observed_at_ms
                ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
                """,
                (
                    "durable-change-train-disposition-artifact",
                    "durable-change-train-disposition-raw",
                    "claude-code-session",
                    "/durable-change-train/disposition-probe.jsonl",
                    "coordinator_session_stream",
                    "supported_parseable",
                    "durable change train probe",
                    1_780_000_000_000,
                    1_780_000_000_000,
                ),
            )
            connection.commit()
        manifest_path = root / "dispositions.jsonl"
        manifest_path.write_text(
            json.dumps(
                {
                    "raw_id": "durable-change-train-disposition-raw",
                    "disposition_kind": "terminal_corrupt_input",
                    "detail": "durable change train read-only probe",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report = actuator(root, manifest_path=manifest_path, dry_run=True)
    if getattr(report, "applied", True) or getattr(report, "candidate_count", 0) != 1:
        raise DurableChangeTrainError("raw failure disposition probe did not remain read-only")
    return "validated one raw failure disposition without mutation"


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


def _verify_persisted_live_tier_continuity(
    conn: sqlite3.Connection,
    train: DurableChangeTrain,
    *,
    actual: DurableDatabaseEvidence | None = None,
) -> None:
    """Prove the exact reopened connection still names the persisted durable tier."""
    if train.apply_evidence is None:
        raise DurableChangeTrainError(f"{train.state.value} train lacks post-apply continuity evidence")
    actual = actual or capture_durable_database_evidence(conn, train.tier)
    expected = train.apply_evidence.post
    if actual.user_version != train.target_version:
        raise DurableChangeTrainError(
            f"{train.tier.value} durable tier continuity proof failed; refusing startup initialization/release"
        )
    try:
        _assert_durable_database_continuity(actual, expected, label=train.tier.value, connection=conn)
    except DurableChangeTrainError as exc:
        raise DurableChangeTrainError(
            f"{train.tier.value} durable tier continuity proof failed; refusing startup initialization/release"
        ) from exc


def _historical_schema_evidence(train: DurableChangeTrain) -> DurableFreshDDLParityProof:
    """Return the immutable historical schema proof a forward admission needs."""
    if train.apply_evidence is None or train.proof is None or train.fresh_ddl_parity is None:
        raise DurableChangeTrainError("released train lacks historical schema evidence for forward-version admission")
    historical = train.proof.fresh_ddl_parity
    if (
        historical.tier is not train.tier
        or historical.target_version != train.target_version
        or historical.migrated_version != train.target_version
        or historical.fresh_version != train.target_version
        or not historical.matches
        or historical.missing_objects
        or historical.unexpected_objects
        or historical.changed_objects
        or historical.migrated_inventory_sha256 != train.apply_evidence.post.schema_inventory_sha256
        or historical.fresh_inventory_sha256 != train.fresh_ddl_parity.fresh_inventory_sha256
    ):
        raise DurableChangeTrainError(
            "released train lacks exact historical schema evidence for forward-version admission"
        )
    return historical


def _canonical_schema_inventory(tier: ArchiveTier, target_version: int) -> _migration_runner.DurableSchemaInventory:
    """Construct the canonical object set for one live durable schema version."""
    try:
        normalized_target_version = int(target_version)
    except (TypeError, ValueError) as exc:
        raise DurableChangeTrainError("canonical schema inventory target version must be an integer") from exc
    if isinstance(target_version, bool):
        raise DurableChangeTrainError("canonical schema inventory target version must be an integer")
    registry = getattr(_migration_runner, "ARCHIVE_DDL_BY_TIER", None)
    archive_ddl = registry.get(tier) if isinstance(registry, dict) else None
    if not isinstance(archive_ddl, str):
        raise DurableChangeTrainError(f"no canonical archive DDL is registered for {tier.value}")
    with closing(sqlite3.connect(":memory:")) as fresh:
        fresh.execute("PRAGMA foreign_keys = ON")
        fresh.executescript(archive_ddl)
        fresh.execute(f"PRAGMA user_version = {normalized_target_version}")
        _migration_runner._prepare_fresh_connection_for_target(fresh, tier, normalized_target_version)
        fresh.commit()
        return _migration_runner.capture_durable_schema_inventory(fresh)


def _verify_released_train_live_tier(
    archive_root: Path,
    conn: sqlite3.Connection,
    train: DurableChangeTrain,
    *,
    current_target_version: int | None = None,
    actual_evidence: DurableDatabaseEvidence | None = None,
    integrity_check: tuple[str, ...] | None = None,
    live_inventory: _migration_runner.DurableSchemaInventory | None = None,
    canonical_inventory: _migration_runner.DurableSchemaInventory | None = None,
) -> DurableForwardVersionReceipt | None:
    """Verify a released train remains represented after later trains advance it."""
    if train.apply_evidence is None:
        raise DurableChangeTrainError(f"{train.state.value} train lacks post-apply continuity evidence")
    actual = actual_evidence or capture_durable_database_evidence(conn, train.tier)
    if actual.user_version < train.target_version:
        raise DurableChangeTrainError(
            f"{train.tier.value} durable tier continuity proof failed: live version regressed below released train "
            "target; refusing startup initialization"
        )
    if actual.user_version == train.target_version:
        if train.source_continuity_evidence is not None:
            _validate_source_continuity_refresh_receipt(archive_root, train)
            _assert_durable_database_continuity(
                actual,
                train.source_continuity_evidence,
                label="source continuity refresh",
                connection=conn,
            )
        else:
            _verify_persisted_live_tier_continuity(conn, train, actual=actual)
        return None
    historical = _historical_schema_evidence(train)
    expected_identity = train.apply_evidence.post.archive_identity_digest
    if not _archive_identity_continuity_matches(
        actual.archive_identity_digest,
        expected_identity,
        archive_root,
        train.tier,
    ):
        raise DurableChangeTrainError(
            f"{train.tier.value} durable tier immutable archive identity differs from historical train "
            f"v{train.target_version} after later train advancement"
        )
    if actual.quick_check != ("ok",):
        raise DurableChangeTrainError(
            f"{train.tier.value} durable tier integrity check failed after later train advancement"
        )
    observed_integrity = integrity_check or tuple(str(row[0]) for row in conn.execute("PRAGMA integrity_check"))
    if observed_integrity != ("ok",):
        raise DurableChangeTrainError(
            f"{train.tier.value} durable tier integrity check failed after later train advancement: {observed_integrity}"
        )
    live_inventory = live_inventory or _migration_runner.capture_durable_schema_inventory(conn)
    if live_inventory.sha256 != actual.schema_inventory_sha256:
        raise DurableChangeTrainError(
            f"{train.tier.value} durable tier schema inventory changed during forward admission"
        )
    expected_inventory = canonical_inventory or _canonical_schema_inventory(train.tier, actual.user_version)
    expected_by_ref = {item.object_ref: item for item in expected_inventory.objects}
    live_by_ref = {item.object_ref: item for item in live_inventory.objects}
    missing = sorted(set(expected_by_ref) - set(live_by_ref))
    unexpected = sorted(set(live_by_ref) - set(expected_by_ref))
    changed = sorted(
        object_ref
        for object_ref in set(expected_by_ref) & set(live_by_ref)
        if expected_by_ref[object_ref].definition_sha256 != live_by_ref[object_ref].definition_sha256
    )
    if missing or unexpected or changed:
        raise DurableChangeTrainError(
            f"{train.tier.value} durable tier schema differs from the canonical live version: "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )
    runtime_target = (
        cast(dict[ArchiveTier, int], vars(_migration_runner)["ARCHIVE_VERSION_BY_TIER"])[train.tier]
        if current_target_version is None
        else current_target_version
    )
    if actual.user_version > runtime_target:
        raise DurableChangeTrainError(
            f"{train.tier.value} durable tier version {actual.user_version} is newer than current target "
            f"v{runtime_target}; historical train v{train.target_version} cannot admit it"
        )
    return DurableForwardVersionReceipt(
        tier=train.tier,
        historical_train_id=train.train_id,
        historical_target_version=train.target_version,
        current_target_version=runtime_target,
        observed_live_version=actual.user_version,
        historical_schema_inventory_sha256=historical.migrated_inventory_sha256,
        archive_identity_digest=actual.archive_identity_digest,
    )


def _forward_version_receipt_for_current_tier(
    archive_root: Path,
    conn: sqlite3.Connection,
    tier: ArchiveTier,
    *,
    current_version: int,
    current_target_version: int,
    evidence: _DurableForwardVersionEvidence | None = None,
) -> DurableForwardVersionReceipt | None:
    """Return the newest released historical-train receipt at the live target."""
    manifest_root = archive_root / ".maintenance-state" / "durable-change-trains"
    manifests_by_target = _released_train_manifests_by_target(manifest_root, tier)
    historical = [
        train
        for train in manifests_by_target.values()
        if train.state is DurableChangeTrainState.RELEASED and train.target_version < current_version
    ]
    if tier in DURABLE_MIGRATION_ADOPTION_FLOORS and current_version > DURABLE_MIGRATION_ADOPTION_FLOORS[tier]:
        _require_released_train_chain(
            tier,
            manifests_by_target,
            current_version=current_version,
            floor=_chain_floor(tier, _fresh_durable_bootstrap_versions(archive_root, manifest_root)),
        )
    if not historical:
        return None
    if evidence is None:
        actual = capture_durable_database_evidence(conn, tier)
        evidence = _DurableForwardVersionEvidence(
            actual=actual,
            integrity_check=tuple(str(row[0]) for row in conn.execute("PRAGMA integrity_check")),
            live_inventory=_migration_runner.capture_durable_schema_inventory(conn),
            canonical_inventory=_canonical_schema_inventory(tier, actual.user_version),
        )
    for train in sorted(historical, key=lambda item: item.target_version, reverse=True):
        receipt = _verify_released_train_live_tier(
            archive_root,
            conn,
            train,
            current_target_version=current_target_version,
            actual_evidence=evidence.actual,
            integrity_check=evidence.integrity_check,
            live_inventory=evidence.live_inventory,
            canonical_inventory=evidence.canonical_inventory,
        )
        if receipt is not None:
            return receipt
    return None


def _released_train_manifests_by_target(
    manifest_root: Path,
    tier: ArchiveTier,
) -> dict[int, DurableChangeTrain]:
    """Load one persisted train record per target version for a durable tier."""
    manifests_by_target: dict[int, DurableChangeTrain] = {}
    if not manifest_root.is_dir():
        return manifests_by_target
    for path in sorted(manifest_root.glob(f"{tier.value}-*.json")):
        train = load_durable_change_train_manifest(path)
        if train.target_version in manifests_by_target:
            raise DurableChangeTrainError(
                f"duplicate {tier.value} durable train manifests for target v{train.target_version}"
            )
        manifests_by_target[train.target_version] = train
    return manifests_by_target


def _require_released_train_chain(
    tier: ArchiveTier,
    manifests_by_target: dict[int, DurableChangeTrain],
    *,
    current_version: int,
    floor: int | None = None,
) -> None:
    """Require released, schema-proven evidence for every later version."""
    chain_floor = DURABLE_MIGRATION_ADOPTION_FLOORS[tier] if floor is None else floor
    if chain_floor < DURABLE_MIGRATION_ADOPTION_FLOORS[tier] or chain_floor > current_version:
        raise DurableChangeTrainError(
            f"invalid {tier.value} durable train chain floor v{chain_floor} for live v{current_version}"
        )
    missing_targets = [
        version
        for version in range(chain_floor + 1, current_version + 1)
        if manifests_by_target.get(version) is None
        or manifests_by_target[version].state is not DurableChangeTrainState.RELEASED
    ]
    if missing_targets:
        raise DurableChangeTrainError(
            f"{tier.value} durable forward admission lacks released train evidence for versions "
            f"{missing_targets} from chain floor v{chain_floor} "
            f"through live v{current_version}"
        )
    for version in range(chain_floor + 1, current_version + 1):
        _historical_schema_evidence(manifests_by_target[version])


def _chain_floor(tier: ArchiveTier, bootstrap_versions: dict[ArchiveTier, int]) -> int:
    """Return the durable train floor allowed by adoption and bootstrap evidence."""
    adoption_floor = DURABLE_MIGRATION_ADOPTION_FLOORS[tier]
    return max(adoption_floor, bootstrap_versions.get(tier, adoption_floor))


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
    """Execute every persisted train state while the caller holds archive ownership.

    The caller-held lease must cover startup reconciliation and receipt creation so
    reused forward-version evidence cannot become stale between those operations.
    """
    forward_version_evidence: dict[ArchiveTier, _DurableForwardVersionEvidence] = {}
    reconcile_durable_change_train_startup(archive_root, live_evidence_cache=forward_version_evidence)
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
        with _open_existing_tier(tier_path) as live:
            forward_version_receipt = _forward_version_receipt_for_current_tier(
                archive_root,
                live,
                tier,
                current_version=current_version,
                current_target_version=runtime_target_version,
                evidence=forward_version_evidence.get(tier),
            )
        return DurableChangeTrainExecution(
            train=None,
            manifest_path=None,
            migration_result=legacy_result,
            forward_version_receipt=forward_version_receipt,
        )

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
            forward_version_receipt = _verify_released_train_live_tier(
                archive_root,
                live,
                train,
                current_target_version=runtime_target_version,
                actual_evidence=(forward_version_evidence[tier].actual if tier in forward_version_evidence else None),
                integrity_check=(
                    forward_version_evidence[tier].integrity_check if tier in forward_version_evidence else None
                ),
                live_inventory=(
                    forward_version_evidence[tier].live_inventory if tier in forward_version_evidence else None
                ),
                canonical_inventory=(
                    forward_version_evidence[tier].canonical_inventory if tier in forward_version_evidence else None
                ),
            )
        return DurableChangeTrainExecution(
            train=train,
            manifest_path=manifest_path,
            migration_result=None,
            forward_version_receipt=forward_version_receipt,
        )

    if train.state is DurableChangeTrainState.DECLARED:
        previous_revision = train.revision
        train = admit_durable_change_train(
            train,
            observed_current_version=current_version,
            fresh_ddl_parity=_fresh_ddl_parity_for_train(train),
            admission_evidence_ref=f"proof:maintenance-admission:{train.train_id}",
            migration_claims=(sidecar.train.migration,),
            # A durable archive may be several numbered slots behind the
            # shipped package.  Admit the exact next train before advancing
            # to a later sidecar.  Comparing a historical slot with the
            # package's final target rejects valid sequential recovery as
            # "stale" before the migration can run.
            canonical_target_version=sidecar.slot,
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


def reconcile_durable_change_train_startup(
    archive_root: Path,
    *,
    live_evidence_cache: dict[ArchiveTier, _DurableForwardVersionEvidence] | None = None,
) -> tuple[Path, ...]:
    """Reconcile interrupted trains while the caller holds archive ownership.

    The caller-held lease must cover any subsequent use of ``live_evidence_cache``.
    """
    from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation

    with OwnedArchiveLocation.acquire(
        ArchiveLocation.resolve(archive_root),
        owner_id=f"durable-train-recovery:{os.getpid()}",
        allow_reentrant=True,
    ):
        return _reconcile_durable_change_train_startup_locked(
            archive_root,
            live_evidence_cache=live_evidence_cache,
        )


def _reconcile_durable_change_train_startup_locked(
    archive_root: Path,
    *,
    live_evidence_cache: dict[ArchiveTier, _DurableForwardVersionEvidence] | None = None,
) -> tuple[Path, ...]:
    """Reconcile persisted trains while the caller holds archive ownership."""
    _recover_pending_source_continuity_intents(archive_root)
    manifest_root = archive_root / ".maintenance-state" / "durable-change-trains"
    reconciled: list[Path] = []
    live_evidence_by_tier: dict[ArchiveTier, DurableDatabaseEvidence] = {}
    live_integrity_by_tier: dict[ArchiveTier, tuple[str, ...]] = {}
    live_inventory_by_tier: dict[ArchiveTier, _migration_runner.DurableSchemaInventory] = {}
    canonical_inventory_by_tier: dict[ArchiveTier, _migration_runner.DurableSchemaInventory] = {}
    manifests_by_tier: dict[ArchiveTier, dict[int, DurableChangeTrain]] = {}
    validated_tiers: set[ArchiveTier] = set()
    manifest_paths = tuple(sorted(manifest_root.glob("*.json")))
    fresh_bootstrap_versions = _fresh_durable_bootstrap_versions(archive_root, manifest_root)

    def record_reconciled(path: Path) -> None:
        if path not in reconciled:
            reconciled.append(path)

    # Recover every non-released lifecycle state before validating any
    # released train. A later committed train may be persisted as
    # backup-authorized, applied, or proven while an older released manifest
    # is still present. Checking the older train first would reject the
    # incomplete chain before startup had a chance to finish that recovery.
    for manifest_path in manifest_paths:
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
            record_reconciled(manifest_path)
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
            record_reconciled(manifest_path)

        if train.state in {
            DurableChangeTrainState.APPLIED,
            DurableChangeTrainState.PROVEN,
        }:
            _prove_and_release_persisted_train(archive_root, manifest_path, train)
            record_reconciled(manifest_path)

    # An existing durable tier above its adoption floor must have a complete
    # released-train chain even when the manifest directory is absent or empty.
    # Fresh database creation does not enter this startup reconciliation route.
    # Recovery runs first so an indeterminate persisted failure keeps its
    # stronger fail-closed error rather than being masked by chain validation.
    for tier, adoption_floor in DURABLE_MIGRATION_ADOPTION_FLOORS.items():
        tier_path = archive_root / f"{tier.value}.db"
        if not tier_path.is_file():
            continue
        with _open_existing_tier(tier_path) as live:
            current_version = int(live.execute("PRAGMA user_version").fetchone()[0] or 0)
        if current_version <= adoption_floor:
            continue
        manifests_by_tier[tier] = _released_train_manifests_by_target(manifest_root, tier)
        tier_manifest_paths = tuple(manifest_root.glob(f"{tier.value}-*.json"))
        bootstrap_version = fresh_bootstrap_versions.get(tier)
        if bootstrap_version is not None and current_version < bootstrap_version:
            raise DurableChangeTrainError(
                f"{tier.value} durable tier regressed below fresh bootstrap v{bootstrap_version}"
            )
        if bootstrap_version == current_version and not tier_manifest_paths:
            validated_tiers.add(tier)
            continue
        _require_released_train_chain(
            tier,
            manifests_by_tier[tier],
            current_version=current_version,
            floor=_chain_floor(tier, fresh_bootstrap_versions),
        )
        validated_tiers.add(tier)

    for manifest_path in manifest_paths:
        train = load_durable_change_train_manifest(manifest_path)
        if train.state is not DurableChangeTrainState.RELEASED:
            continue
        with _open_existing_tier(archive_root / f"{train.tier.value}.db") as live:
            actual = live_evidence_by_tier.get(train.tier)
            if actual is None:
                actual = capture_durable_database_evidence(live, train.tier)
                live_evidence_by_tier[train.tier] = actual
            if (
                actual.user_version > DURABLE_MIGRATION_ADOPTION_FLOORS[train.tier]
                and train.tier not in validated_tiers
            ):
                if train.tier not in manifests_by_tier:
                    manifests_by_tier[train.tier] = _released_train_manifests_by_target(manifest_root, train.tier)
                _require_released_train_chain(
                    train.tier,
                    manifests_by_tier[train.tier],
                    current_version=actual.user_version,
                    floor=_chain_floor(train.tier, fresh_bootstrap_versions),
                )
            if actual.user_version > train.target_version:
                if train.tier not in live_integrity_by_tier:
                    live_integrity_by_tier[train.tier] = tuple(
                        str(row[0]) for row in live.execute("PRAGMA integrity_check")
                    )
                if train.tier not in live_inventory_by_tier:
                    live_inventory_by_tier[train.tier] = _migration_runner.capture_durable_schema_inventory(live)
                if train.tier not in canonical_inventory_by_tier:
                    canonical_inventory_by_tier[train.tier] = _canonical_schema_inventory(
                        train.tier, actual.user_version
                    )
                if live_evidence_cache is not None:
                    live_evidence_cache[train.tier] = _DurableForwardVersionEvidence(
                        actual=actual,
                        integrity_check=live_integrity_by_tier[train.tier],
                        live_inventory=live_inventory_by_tier[train.tier],
                        canonical_inventory=canonical_inventory_by_tier[train.tier],
                    )
            _verify_released_train_live_tier(
                archive_root,
                live,
                train,
                actual_evidence=actual,
                integrity_check=live_integrity_by_tier.get(train.tier),
                live_inventory=live_inventory_by_tier.get(train.tier),
                canonical_inventory=canonical_inventory_by_tier.get(train.tier),
            )
        record_reconciled(manifest_path)
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
    "DurableForwardVersionReceipt",
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
    "DurableSourceTrainMissingError",
    "DurableSourceContinuitySemanticError",
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
    "assert_source_continuity_apply_allowed",
    "write_source_continuity_pending_intent",
    "mark_source_continuity_pending_intent_terminal",
    "clear_source_continuity_pending_intent",
    "refresh_released_source_train_continuity",
    "write_durable_change_train_manifest",
    "load_durable_change_train_manifest",
]

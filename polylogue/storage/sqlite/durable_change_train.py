"""Durable source/user/audit migration change-train authority."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import re
import sqlite3
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Final, cast

import polylogue.storage.sqlite.migration_runner as _migration_runner
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_FORMAT_FLOOR_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.managed_connection import sqlite_connection
from polylogue.storage.sqlite.migration_runner import (
    DURABLE_CHANGE_TRAIN_FORMAT,
    DurableChangeTrain,
    DurableChangeTrainApplyError,
    DurableChangeTrainError,
    DurableChangeTrainRecoveryError,
    DurableChangeTrainState,
    DurableDatabaseEvidence,
    DurableMigrationClaim,
    DurableMigrationReplayProof,
    DurableRuntimeConsumerResult,
    DurableTierNewerThanRuntimeError,
    MigrationResult,
    _assert_durable_database_continuity,
    _canonical_json_sha256,
    _validate_riders,
    add_durable_change_train_rider,
    admit_durable_change_train,
    apply_durable_change_train,
    authorize_durable_change_train_backup,
    capture_durable_database_evidence,
    capture_durable_schema_inventory,
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
    rehearse_durable_migration_chain,
    release_durable_change_train,
    reserve_durable_change_train,
    validate_durable_change_train_manifest,
    validate_durable_migration_replay_proof,
    write_durable_change_train_manifest,
)

if TYPE_CHECKING:
    from polylogue.security.excision_policy import ExcisionPolicySnapshot

DURABLE_MIGRATION_ADOPTION_FLOORS: Final[dict[ArchiveTier, int]] = {
    # Numbering was reset for the marker-identified archive lineage.  A
    # future train starts directly above this floor; historical chains are
    # not a bridge into the new format.
    ArchiveTier.SOURCE: ARCHIVE_FORMAT_FLOOR_VERSION,
    ArchiveTier.USER: ARCHIVE_FORMAT_FLOOR_VERSION,
    ArchiveTier.AUDIT: ARCHIVE_FORMAT_FLOOR_VERSION,
}
_SIDECAR_NAME_RE = re.compile(r"^(?P<slot>\d{3,})\.train\.json$")
_DURABLE_TRAIN_MANIFEST_NAME_RE = re.compile(r"^(?P<tier>source|user|audit)-(?P<slot>\d{3,})\.json$")
_MIGRATION_NAME_RE = re.compile(r"^(?P<slot>\d{3,})_[a-z0-9_]+\.sql$")
_FRESH_DURABLE_BOOTSTRAP_FORMAT = "polylogue.durable-bootstrap.v1"
_FRESH_DURABLE_BOOTSTRAP_MARKER = ".bootstrap"


def durable_train_manifest_paths(manifest_root: Path, tier: ArchiveTier | None = None) -> tuple[Path, ...]:
    """Return only positively typed durable-train entries.

    Durable train authority is identified by its tier and numeric train slot.
    Other maintenance receipts may live alongside the train state, but are
    never handed to the train parser.  In particular, this is an allow-shaped
    ownership rule rather than a denylist of receipt filenames.
    """
    if not manifest_root.is_dir():
        return ()
    prefix = f"{tier.value}-" if tier is not None else ""
    paths: list[Path] = []
    for path in sorted(manifest_root.glob(f"{prefix}*.json")):
        match = _DURABLE_TRAIN_MANIFEST_NAME_RE.fullmatch(path.name)
        if match is None or (tier is not None and match.group("tier") != tier.value):
            continue
        paths.append(path)
    return tuple(paths)


_FRESH_DURABLE_BOOTSTRAP_PENDING_MARKER = ".bootstrap.pending"


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
class _ReleasedSchemaEvidence:
    """Live released admission deliberately excludes mutable row inventories."""

    user_version: int
    quick_check: tuple[str, ...]
    schema_inventory_sha256: str
    archive_identity_digest: str


def _capture_released_schema_evidence(conn: sqlite3.Connection, tier: ArchiveTier) -> _ReleasedSchemaEvidence:
    inventory = _migration_runner.capture_durable_schema_inventory(conn)
    return _ReleasedSchemaEvidence(
        user_version=int(conn.execute("PRAGMA user_version").fetchone()[0] or 0),
        quick_check=tuple(str(row[0]) for row in conn.execute("PRAGMA quick_check")),
        schema_inventory_sha256=inventory.sha256,
        archive_identity_digest=_migration_runner._durable_archive_identity_digest(conn, tier),
    )


@dataclass(frozen=True, slots=True)
class _DurableForwardVersionEvidence:
    """Cached live evidence reused by one no-op maintenance execution."""

    actual: _ReleasedSchemaEvidence
    integrity_check: tuple[str, ...]
    live_inventory: _migration_runner.DurableSchemaInventory
    expected_live_schema_inventory_sha256: str


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
    # The canonical classifier separately proves paired index replacement;
    # destructive drops still require their declared copy-forward constraints.
    if (
        _migration_runner._migration_has_schema_drops(sql)
        and expected_claim.requires_backup
        and not train.drop_constraints
    ):
        try:
            _migration_runner._index_replacement_pairs(Path(migration_name), sql, allow_other_statements=True)
        except _migration_runner.MigrationError as exc:
            raise DurableChangeTrainError(
                f"durable migration sidecar forbids an unapproved drop: {sidecar.resource_name}"
            ) from exc


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
    """Record the versions a direct, current-schema bootstrap created.

    The marker records what the bootstrap produced, not where it produced it.
    Binding it to the archive root path and to the source/user inodes -- as
    earlier revisions did -- made an ordinary ``mv`` of the archive root, a
    restore from backup, or a cross-filesystem move a permanent refusal with
    no sanctioned repair.  Authenticity is re-established on read from the
    archive's own durable content (``_assert_fresh_durable_bootstrap_is_own``).
    """
    archive_root = archive_root.resolve()
    marker_root = archive_root / ".maintenance-state" / "durable-change-trains"
    marker_path = marker_root / _FRESH_DURABLE_BOOTSTRAP_MARKER
    pending_path = marker_root / _FRESH_DURABLE_BOOTSTRAP_PENDING_MARKER
    if marker_path.exists() or durable_train_manifest_paths(marker_root):
        raise DurableChangeTrainError(f"cannot record fresh durable bootstrap over existing train state: {marker_root}")
    if pending_path.is_file():
        _validate_fresh_durable_bootstrap_intent(archive_root)
    marker_root.mkdir(parents=True, exist_ok=True)
    versions: dict[str, int] = {}
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    for tier in DURABLE_MIGRATION_ADOPTION_FLOORS:
        with closing(open_readonly_connection(archive_root / f"{tier.value}.db", validate_schema=False)) as connection:
            versions[tier.value] = int(connection.execute("PRAGMA user_version").fetchone()[0])
    payload: dict[str, object] = {
        "format": _FRESH_DURABLE_BOOTSTRAP_FORMAT,
        "versions": versions,
    }
    payload["marker_digest"] = _bootstrap_marker_digest(payload)
    _write_bootstrap_receipt(marker_path, payload)
    pending_path.unlink(missing_ok=True)


def _record_fresh_durable_bootstrap_intent(archive_root: Path) -> None:
    """Record an authenticated intent before creating the first tier file.

    Fresh archive initialization creates several independent SQLite files. A
    failure in a later tier can therefore leave a partial fresh archive. The
    intent distinguishes that recoverable state from an established archive
    whose durable train evidence has been lost.
    """
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_BASELINE_VERSION_BY_TIER

    archive_root = archive_root.resolve()
    marker_root = archive_root / ".maintenance-state" / "durable-change-trains"
    marker_path = marker_root / _FRESH_DURABLE_BOOTSTRAP_MARKER
    pending_path = marker_root / _FRESH_DURABLE_BOOTSTRAP_PENDING_MARKER
    if marker_path.exists() or durable_train_manifest_paths(marker_root):
        raise DurableChangeTrainError(
            f"cannot record fresh durable bootstrap intent over existing train state: {marker_root}"
        )
    if pending_path.is_file():
        _validate_fresh_durable_bootstrap_intent(archive_root)
        return
    versions = {tier.value: ARCHIVE_BASELINE_VERSION_BY_TIER[tier] for tier in DURABLE_MIGRATION_ADOPTION_FLOORS}
    payload: dict[str, object] = {
        "format": _FRESH_DURABLE_BOOTSTRAP_FORMAT,
        "state": "pending",
        "durable_identity_digest": _fresh_bootstrap_intent_identity_digest(archive_root),
        "versions": versions,
    }
    payload["marker_digest"] = _bootstrap_marker_digest(payload)
    _write_bootstrap_receipt(pending_path, payload)


def _validate_fresh_durable_bootstrap_intent(archive_root: Path) -> None:
    """Validate the authenticated intent for a recoverable fresh bootstrap."""
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_BASELINE_VERSION_BY_TIER

    archive_root = archive_root.resolve()
    marker_path = (
        archive_root / ".maintenance-state" / "durable-change-trains" / _FRESH_DURABLE_BOOTSTRAP_PENDING_MARKER
    )
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DurableChangeTrainError(f"invalid fresh durable bootstrap intent: {marker_path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != _FRESH_DURABLE_BOOTSTRAP_FORMAT:
        raise DurableChangeTrainError(f"fresh durable bootstrap intent format mismatch: {marker_path}")
    if payload.get("state") != "pending":
        raise DurableChangeTrainError(f"fresh durable bootstrap intent state is invalid: {marker_path}")
    if payload.get("durable_identity_digest") != _fresh_bootstrap_intent_identity_digest(archive_root):
        raise DurableChangeTrainError("fresh durable bootstrap intent durable identity mismatch")
    marker_digest = payload.get("marker_digest")
    unsigned_payload = dict(payload)
    unsigned_payload.pop("marker_digest", None)
    if not isinstance(marker_digest, str) or marker_digest != _bootstrap_marker_digest(unsigned_payload):
        raise DurableChangeTrainError("fresh durable bootstrap intent digest mismatch")
    raw_versions = payload.get("versions")
    if not isinstance(raw_versions, dict):
        raise DurableChangeTrainError(f"fresh durable bootstrap intent versions are invalid: {marker_path}")
    for tier in DURABLE_MIGRATION_ADOPTION_FLOORS:
        if raw_versions.get(tier.value) != ARCHIVE_BASELINE_VERSION_BY_TIER[tier]:
            raise DurableChangeTrainError(f"fresh durable bootstrap intent target version is stale: {marker_path}")


def _write_bootstrap_receipt(marker_path: Path, payload: dict[str, object]) -> None:
    """Atomically publish one bootstrap receipt and fsync its directory."""
    marker_root = marker_path.parent
    marker_root.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=marker_root,
            prefix=f".{marker_path.name}.",
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


def _fresh_bootstrap_intent_identity_digest(archive_root: Path) -> str:
    """Bind a pre-file bootstrap intent to its root without inode identity."""
    return _canonical_json_sha256(
        {
            "configured_root": str(archive_root.resolve().absolute()),
            "purpose": "fresh-durable-bootstrap",
        }
    )


def _released_train_proven_floor(manifest_root: Path, tier: ArchiveTier) -> int | None:
    """Return the version a released train proved this tier stood at before it ran.

    Every released train records the live pre-apply evidence it migrated from:
    the tier's ``user_version`` and its schema inventory digest, captured on
    this archive under backup authorization.  When that pre-state matches the
    canonical DDL for its version, the train is the archive's own proof that
    the tier legitimately reached that version -- the same fact the fresh
    bootstrap marker asserts.  This is what lets the marker be retired.
    """
    manifests = _released_train_manifests_by_target(manifest_root, tier)
    for _target, train in sorted(manifests.items()):
        if getattr(train, "state", None) is not DurableChangeTrainState.RELEASED:
            continue
        # Attribute-wise rather than field-wise: a train whose manifest predates
        # apply evidence, or a partially specified stand-in, simply proves
        # nothing here and must not turn an unrelated startup into an error.
        pre = getattr(getattr(train, "apply_evidence", None), "pre", None)
        current_version = getattr(train, "current_version", None)
        pre_version = getattr(pre, "user_version", None)
        if not isinstance(pre_version, int) or pre_version != current_version:
            continue
        if pre_version <= DURABLE_MIGRATION_ADOPTION_FLOORS[tier]:
            # At or below the adoption floor this proves no authority the floor
            # does not already grant, and the canonical image for a pre-floor
            # version is not reconstructible from the shipped migration train.
            continue
        try:
            canonical = _canonical_schema_inventory(tier, pre_version).sha256
        except DurableChangeTrainError:
            # A version whose canonical image cannot be reconstructed proves
            # nothing here; it must not turn an unrelated read into an error.
            continue
        if getattr(pre, "schema_inventory_sha256", None) != canonical:
            continue
        return pre_version
    return None


def _fresh_durable_bootstrap_tier_is_own(
    archive_root: Path,
    manifest_root: Path,
    tier: ArchiveTier,
    version: int,
) -> bool:
    """Decide whether this archive's own content corroborates one marker claim.

    Two proofs are accepted, neither of which mentions the archive root path or
    any inode, so both survive ``mv``, a restore from backup, and a
    cross-filesystem move:

    * the live tier still stands at the recorded version with exactly the
      canonical schema for it -- the archive is materially the bootstrap the
      marker describes; or
    * a released train on this archive proved the tier stood at that version
      before it migrated away from it.
    """
    tier_path = archive_root / f"{tier.value}.db"
    if tier_path.is_file():
        with _open_existing_tier(tier_path) as connection:
            live_version = int(connection.execute("PRAGMA user_version").fetchone()[0] or 0)
            if live_version == version:
                inventory = _migration_runner.capture_durable_schema_inventory(connection)
                if inventory.sha256 == _canonical_schema_inventory(tier, version).sha256:
                    return True
    proven = _released_train_proven_floor(manifest_root, tier)
    return proven is not None and proven >= version


def _fresh_durable_bootstrap_tier_version_skew(archive_root: Path, tier: ArchiveTier, version: int) -> bool:
    """Report whether a live tier simply stands at a different version.

    A tier whose own ``user_version`` disagrees with the marker is neither
    corroboration nor proof of a transplanted marker: it is ordinary durable
    schema skew, which the startup mismatch gate already owns by leaving the
    daemon degraded. Treating it as tampering turns that designed park back
    into the startup crash polylogue-39pdi fixed.
    """
    tier_path = archive_root / f"{tier.value}.db"
    if not tier_path.is_file():
        return False
    with _open_existing_tier(tier_path) as connection:
        live_version = int(connection.execute("PRAGMA user_version").fetchone()[0] or 0)
    return live_version != version


def _assert_fresh_durable_bootstrap_is_own(
    archive_root: Path,
    manifest_root: Path,
    versions: dict[ArchiveTier, int],
) -> set[ArchiveTier]:
    """Refuse a bootstrap marker this archive's own durable content denies.

    Ownership is proved only from the archive's durable content. Markers
    written by earlier revisions may still carry a path-and-inode
    ``durable_identity_digest`` seal; it is ignored, because honouring it
    short-circuited the released-train proof and refused a tier that had
    legitimately migrated away from its bootstrap version (polylogue-zukcl).
    """
    ungranted: set[ArchiveTier] = set()
    for tier, version in versions.items():
        if version <= DURABLE_MIGRATION_ADOPTION_FLOORS[tier]:
            # The marker grants nothing above the adoption floor for this tier.
            continue
        if _fresh_durable_bootstrap_tier_is_own(archive_root, manifest_root, tier, version):
            continue
        if _fresh_durable_bootstrap_tier_version_skew(archive_root, tier, version):
            # Skew, not transplantation: grant this tier nothing and let the
            # ordinary mismatch path decide, rather than failing startup.
            ungranted.add(tier)
            continue
        raise DurableChangeTrainError(
            f"fresh durable bootstrap marker is not this archive's own {tier.value} bootstrap evidence"
        )
    return ungranted


def _fresh_durable_bootstrap_versions(archive_root: Path, marker_root: Path) -> dict[ArchiveTier, int]:
    """Return direct-bootstrap versions when the marker is authentic."""
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
    ungranted = _assert_fresh_durable_bootstrap_is_own(archive_root, marker_root, versions)
    return {tier: version for tier, version in versions.items() if tier not in ungranted}


def _bootstrap_marker_digest(payload: dict[str, object]) -> str:
    """Authenticate bootstrap identity and recorded durable versions together."""
    return _canonical_json_sha256(payload)


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


_ISOLATED_RUNTIME_PROBE_CACHE: dict[tuple[str, Callable[..., object]], str] = {}
_ISOLATED_RUNTIME_PROBE_CACHE_LOCK = Lock()


def _runtime_consumer_results(
    train: DurableChangeTrain,
    archive_root: Path,
    *,
    candidate: sqlite3.Connection,
) -> tuple[DurableRuntimeConsumerResult, ...]:
    """Bind isolated probes to the already authenticated post-apply candidate."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import (
        RuntimeTierProbeAuthority,
        runtime_tier_probe_authority,
    )

    replay = train.schema_replay_proof
    if replay is None:
        raise DurableChangeTrainError("runtime consumer probes require the numbered schema replay authority")
    validate_durable_migration_replay_proof(replay, recompute_installed_bindings=True)
    step = _migration_runner._durable_migration_replay_step(replay, train.target_version)
    inventory = capture_durable_schema_inventory(candidate)
    version = int(candidate.execute("PRAGMA user_version").fetchone()[0])
    if version != train.target_version or inventory.sha256 != step.after_schema_inventory_sha256:
        raise DurableChangeTrainError("runtime consumer candidate differs from its post-apply schema/version authority")
    with runtime_tier_probe_authority(RuntimeTierProbeAuthority(train.tier, version, inventory.sha256)):
        results = _invoke_runtime_consumers(train, archive_root)
    if (
        int(candidate.execute("PRAGMA user_version").fetchone()[0]) != version
        or capture_durable_schema_inventory(candidate).sha256 != inventory.sha256
    ):
        raise DurableChangeTrainError("runtime consumer candidate changed during its isolated probes")
    return results


def _invoke_runtime_consumers(
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
                    with sqlite_connection(":memory:") as probe:
                        value(probe, train.tier)
                elif reference.endswith(":write_source_blob_refs"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError("attachment reference writer requires Source")
                    detail = _probe_attachment_coordinate_writer(cast(Callable[..., object], value))
                elif reference.endswith(":write_source_hook_event"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_source_hook_event_writer(cast(Callable[..., object], value))
                elif reference.endswith(":read_frontier_journal_state") or reference.endswith(
                    ":read_frontier_inspection_mark"
                ):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError("frontier dependency probes require the Source train")
                    detail = _probe_frontier_dependency_read(
                        cast(Callable[..., object], value), journal=reference.endswith(":read_frontier_journal_state")
                    )
                elif reference.endswith(":record_raw_profile_identity") or reference.endswith(
                    ":read_raw_profile_identity"
                ):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_source_profile_identity(
                        cast(Callable[..., object], value), writer=reference.endswith(":record_raw_profile_identity")
                    )
                elif reference.endswith(":has_raw_byte_revision_dependents"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError("byte revision dependency reader is source-tier-only")
                    detail = _probe_raw_byte_revision_dependents(cast(Callable[..., object], value))
                elif reference.endswith(":read_raw_failure_lifecycle"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    with _open_existing_tier(archive_root / "source.db") as live:
                        preparation_key = _migration_runner.durable_preparation_fingerprint(
                            live, train.tier, consumer_paths=(module_ref,)
                        )
                    key = (preparation_key, cast(Callable[..., object], value))
                    with _ISOLATED_RUNTIME_PROBE_CACHE_LOCK:
                        cached_detail = _ISOLATED_RUNTIME_PROBE_CACHE.get(key)
                    if cached_detail is None:
                        cached_detail = _probe_raw_failure_lifecycle(cast(Callable[..., object], value), archive_root)
                        with _ISOLATED_RUNTIME_PROBE_CACHE_LOCK:
                            if len(_ISOLATED_RUNTIME_PROBE_CACHE) >= 64:
                                _ISOLATED_RUNTIME_PROBE_CACHE.pop(next(iter(_ISOLATED_RUNTIME_PROBE_CACHE)))
                            _ISOLATED_RUNTIME_PROBE_CACHE[key] = cached_detail
                    detail = cached_detail
                elif reference.endswith(":execute_raw_admission_plan_sync"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_captured_zip_admission(cast(Callable[..., object], value))
                elif reference.endswith(":retained_source_location"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError("captured ZIP restoration requires source tier")
                    detail = _probe_captured_zip_restoration(cast(Callable[..., object], value))
                elif reference.endswith(":raw_revision_descriptor"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_revision_provider_resolution(cast(Callable[..., object], value))
                elif reference.endswith(":_row_to_raw_session"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_raw_record_hydration(cast(Callable[..., object], value))
                elif reference.endswith(":upsert_raw_artifact"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_raw_artifact_upsert(cast(Callable[..., object], value))
                elif reference.endswith(":AuditRepository.reconcile_continuity"):
                    from polylogue.operations.audit import AuditRepository

                    AuditRepository.for_archive_root(archive_root).reconcile_continuity()
                    detail = "reconciled matching source/audit continuity heads"
                elif reference.endswith(":compile_raw_state_update"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_raw_state_update_compile(cast(Callable[..., object], value))
                elif reference.endswith((":append_accepted_marker_input", ":read_accepted_marker_inputs")):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError("accepted marker input writer requires source tier")
                    detail = _probe_accepted_marker_input_writer()
                elif reference.endswith(":AuditContinuityCoordinator"):
                    from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator

                    detail = AuditContinuityCoordinator(archive_root).runtime_probe()
                elif reference.endswith(":publish_source_generation"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = (
                        _probe_captured_source_input(cast(Callable[..., object], value), writer=True)
                        if consumer.consumer_id == "captured-input-publication"
                        else _probe_source_generation_publish(cast(Callable[..., object], value))
                    )
                elif reference.endswith(":page_retained_source_inputs"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_captured_source_input(cast(Callable[..., object], value), writer=False)
                elif reference.endswith(":read_excision_policy_projection"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_excision_policy_projection_read(cast(Callable[..., object], value))
                elif reference.endswith(":record_source_attachments"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_source_attachment_record(cast(Callable[..., object], value))
                elif reference.endswith(":source_attachment_census"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_source_attachment_census(cast(Callable[..., object], value))
                elif reference.endswith(":source_generation_census"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_source_generation_census(cast(Callable[..., object], value))
                elif reference.endswith(":admit_material"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_material_admission(cast(Callable[..., object], value))
                elif reference.endswith(":get_material"):
                    if train.tier is not ArchiveTier.SOURCE:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is source-tier-only: {reference}"
                        )
                    detail = _probe_material_read(cast(Callable[..., object], value))
                elif reference.endswith(":promote_query"):
                    if train.tier is not ArchiveTier.USER:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is user-tier-only: {reference}"
                        )
                    detail = _probe_query_promotion(cast(Callable[..., object], value))
                elif reference.endswith(":apply_query_excision"):
                    if train.tier is not ArchiveTier.USER:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is user-tier-only: {reference}"
                        )
                    detail = _probe_query_excision(cast(Callable[..., object], value))
                elif reference.endswith(":upsert_assertion"):
                    if train.tier is not ArchiveTier.USER:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is user-tier-only: {reference}"
                        )
                    detail = _probe_assertion_upsert(cast(Callable[..., object], value))
                elif reference.endswith(":mark_assertion_status"):
                    if train.tier is not ArchiveTier.USER:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is user-tier-only: {reference}"
                        )
                    detail = _probe_assertion_status_mark(cast(Callable[..., object], value))
                elif reference.endswith(":advance_session_marker_delivery"):
                    if train.tier is not ArchiveTier.USER:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is user-tier-only: {reference}"
                        )
                    detail = _probe_session_marker_delivery_writer(cast(Callable[..., object], value))
                elif reference.endswith(":advance_accepted_marker_delivery_cursor"):
                    if train.tier is not ArchiveTier.USER:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is user-tier-only: {reference}"
                        )
                    detail = _probe_accepted_marker_delivery_cursor_writer(cast(Callable[..., object], value))
                elif reference.endswith(":accepted_marker_delivery_cursor"):
                    if train.tier is not ArchiveTier.USER:
                        raise DurableChangeTrainError(
                            f"runtime consumer {consumer.consumer_id} is user-tier-only: {reference}"
                        )
                    detail = _probe_accepted_marker_delivery_cursor_reader(cast(Callable[..., object], value))
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


def _probe_attachment_coordinate_writer(writer: Callable[..., object]) -> str:
    """Prove equal bytes keep distinct provider coordinates and raw ownership."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_runtime_tier_probe
    from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceBlobRef

    with sqlite_connection(":memory:") as probe:
        initialize_runtime_tier_probe(probe, ArchiveTier.SOURCE)
        blob_hash = hashlib.sha256(b"attachment-coordinate-probe").digest()
        writer(
            probe,
            "attachment-coordinate-raw",
            lambda: (
                ArchiveSourceBlobRef(
                    blob_hash=blob_hash,
                    ref_type="attachment",
                    source_path=f"attachment:{file_id}",
                    size_bytes=27,
                    acquired_at_ms=1,
                )
                for file_id in ("file-a", "file-b")
            ),
        )
        rows = probe.execute("SELECT ref_id, source_path, blob_hash FROM blob_refs ORDER BY source_path").fetchall()
        if rows != [
            ("attachment-coordinate-raw", "attachment:file-a", blob_hash),
            ("attachment-coordinate-raw", "attachment:file-b", blob_hash),
        ]:
            raise DurableChangeTrainError("attachment writer collapsed equal-content coordinates")
    return "retained two equal-content attachment coordinates under their original raw owner"


def _probe_source_hook_event_writer(writer: Callable[..., object]) -> str:
    """Exercise the source hook writer against an isolated fresh source tier."""
    from polylogue.core.enums import Origin
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_runtime_tier_probe
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
    with sqlite_connection(":memory:") as probe:
        initialize_runtime_tier_probe(probe, ArchiveTier.SOURCE)
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


def _probe_raw_artifact_upsert(upsert: Callable[..., object]) -> str:
    """Exercise raw-artifact admission against the canonical source schema."""
    from polylogue.core.enums import ArtifactSupportStatus, Origin
    from polylogue.storage.sqlite.archive_tiers.source_write import (
        ArchiveSourceArtifact,
        deterministic_blob_hash,
        write_source_raw_session_blob_ref,
    )

    source_path = "/durable-change-train/raw-artifact-probe.jsonl"
    payload = b'{"probe":"raw-artifact"}'
    raw_id = "durable-change-train-raw-artifact-raw"
    blob_hash = deterministic_blob_hash(payload)
    artifact = ArchiveSourceArtifact(
        artifact_id="durable-change-train-raw-artifact",
        origin=Origin.CODEX_SESSION,
        source_path=source_path,
        source_index=0,
        artifact_kind="agent_transcript",
        classification_reason="durable-change-train probe",
        support_status=ArtifactSupportStatus.SUPPORTED_PARSEABLE,
        first_observed_at_ms=1_780_000_000_000,
        last_observed_at_ms=1_780_000_000_000,
    )
    with _runtime_probe_source_connection() as probe:
        write_source_raw_session_blob_ref(
            probe,
            origin=Origin.CODEX_SESSION,
            source_path=source_path,
            canonical_source_path=source_path,
            source_index=0,
            blob_hash=blob_hash,
            blob_size=len(payload),
            acquired_at_ms=1_780_000_000_000,
            raw_id=raw_id,
        )
        upsert(probe, raw_id, artifact)
        row = probe.execute(
            "SELECT artifact_id, raw_id, origin, source_path, source_index, artifact_kind, support_status "
            "FROM raw_artifacts"
        ).fetchone()
    expected = (
        artifact.artifact_id,
        raw_id,
        Origin.CODEX_SESSION.value,
        source_path,
        0,
        artifact.artifact_kind,
        ArtifactSupportStatus.SUPPORTED_PARSEABLE.value,
    )
    if row != expected:
        raise DurableChangeTrainError("raw-artifact upsert probe did not persist the expected artifact contract")
    return "wrote and read back one raw artifact in the canonical source tier"


def _probe_accepted_marker_input_writer() -> str:
    import asyncio

    import aiosqlite

    from polylogue.storage.accepted_marker_inputs import (
        AcceptedMarkerInputRefusedError,
        append_accepted_marker_input,
        prepare_accepted_marker_input,
        read_accepted_marker_inputs,
    )
    from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL

    async def exercise() -> None:
        async with aiosqlite.connect(":memory:") as conn:
            await conn.executescript(SOURCE_DDL)
            batch = prepare_accepted_marker_input("synthetic-marker-input", [{"session_id": "s", "candidates": []}])
            first = await append_accepted_marker_input(conn, batch)
            replay = await append_accepted_marker_input(conn, batch)
            if first != 1 or replay != first:
                raise DurableChangeTrainError("marker input replay advanced the stream")
            page = await read_accepted_marker_inputs(conn, limit=1)
            if (
                len(page) != 1
                or page[0].batch.raw_id != batch.raw_id
                or page[0].batch.identity != batch.identity
                or page[0].batch.payload_sha256 != batch.payload_sha256
                or page[0].sequence != first
            ):
                raise DurableChangeTrainError("marker input reader lost retained bytes or sequence")
            if await read_accepted_marker_inputs(conn, after_sequence=first):
                raise DurableChangeTrainError("marker input pagination repeated its cursor")
            conflict = prepare_accepted_marker_input(
                "synthetic-marker-input", [{"session_id": "s", "candidates": [{"body": "changed"}]}]
            )
            try:
                await append_accepted_marker_input(conn, conflict)
            except AcceptedMarkerInputRefusedError:
                pass
            else:
                raise DurableChangeTrainError("marker input writer accepted conflicting replay")
            await conn.rollback()
            cursor = await conn.execute("SELECT COUNT(*) FROM accepted_marker_inputs")
            count = await cursor.fetchone()
            if count is None or count[0] != 0:
                raise DurableChangeTrainError("marker input survived source rollback")

    asyncio.run(exercise())
    return "accepted marker replay is immutable and source rollback removes the batch"


@contextmanager
def _runtime_probe_source_connection() -> Iterator[sqlite3.Connection]:
    """Own the actual canonical source-tier probe through native settlement."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_runtime_tier_probe

    with sqlite_connection(":memory:") as connection:
        initialize_runtime_tier_probe(connection, ArchiveTier.SOURCE)
        yield connection


def _probe_source_profile_identity(operation: Callable[..., object], *, writer: bool) -> str:
    """Exercise immutable receipt write/read and the historical evidence gap."""
    from polylogue.core.enums import Origin
    from polylogue.storage.sqlite.archive_tiers.source_write import (
        read_raw_profile_identity,
        record_raw_profile_identity,
        write_source_raw_session_blob_ref,
    )

    write = operation if writer else record_raw_profile_identity
    read = read_raw_profile_identity if writer else operation
    key = "0123456789ab"
    with _runtime_probe_source_connection() as probe:
        for raw_id in ("captured-profile", "historical-profile-gap"):
            write_source_raw_session_blob_ref(
                probe,
                origin=Origin.UNKNOWN_EXPORT,
                source_path=f"/durable-change-train/{raw_id}.json",
                canonical_source_path=f"/durable-change-train/{raw_id}.json",
                source_index=0,
                blob_hash=hashlib.sha256(raw_id.encode()).digest(),
                blob_size=len(raw_id),
                acquired_at_ms=1,
                raw_id=raw_id,
            )
        if read(probe, "captured-profile") is not None:
            raise DurableChangeTrainError("profile receipt probe fabricated an absent qualifier")
        write(probe, raw_id="captured-profile", profile_key=key, allow_new_receipt=True)
        write(probe, raw_id="captured-profile", profile_key=key)
        if read(probe, "captured-profile") != key:
            raise DurableChangeTrainError("profile receipt reader changed the accepted qualifier")
        for raw_id, incoming in (("captured-profile", "abcdef012345"), ("historical-profile-gap", key)):
            try:
                write(probe, raw_id=raw_id, profile_key=incoming)
            except ValueError:
                pass
            else:
                raise DurableChangeTrainError("profile receipt writer relabeled retained acquisition evidence")
        if read(probe, "historical-profile-gap") is not None or read(probe, "captured-profile") != key:
            raise DurableChangeTrainError("profile receipt refusal changed durable evidence")
    return "profile receipt round-trips immutably; absent historical evidence remains absent"


def _probe_captured_source_input(operation: Callable[..., object], *, writer: bool) -> str:
    """Invoke both production sides of the installed captured-input columns."""
    from polylogue.core.provider_identity import captured_hermes_profile_key
    from polylogue.storage.sqlite.archive_tiers.source_items import (
        CapturedSourceInputIdentity,
        page_retained_source_inputs,
        publish_source_generation,
    )

    publish = operation if writer else publish_source_generation
    page = page_retained_source_inputs if writer else operation
    root = Path("/durable-change-train/profile")
    identity = CapturedSourceInputIdentity(
        canonical_source_path="/durable-change-train/physical/input.jsonl",
        semantic_source_path=str(root / "sessions/input.jsonl"),
        profile_root=str(root),
        profile_key=captured_hermes_profile_key(root),
        profile_source_path=str(root / "sessions/input.jsonl"),
    )
    generation = "durable-change-train-captured-input"
    with _runtime_probe_source_connection() as probe:
        publish(
            probe,
            source_generation_id=generation,
            manifest_digest="1" * 64,
            addressing_mode="physical-file-v1",
            coordinates=("sessions/input.jsonl",),
            observed_at_ms=1,
            source_paths={"sessions/input.jsonl": identity.semantic_source_path},
            input_blob_hashes={"sessions/input.jsonl": bytes.fromhex("2" * 64)},
            enumeration_fingerprint="3" * 64,
            captured_input_identities={"sessions/input.jsonl": identity},
            policy_snapshot=_probe_excision_policy_snapshot(generation),
        )
        rows = page(probe, generation, limit=1)
        if not isinstance(rows, tuple) or len(rows) != 1 or rows[0][0].captured_identity != identity:
            raise DurableChangeTrainError("captured input lost its accepted semantic/physical/profile receipt")
        if page(probe, generation, after=("sessions/input.jsonl", rows[0][0].source_item_id), limit=1):
            raise DurableChangeTrainError("captured input reader repeated its page cursor")
    return "published and paged the exact captured physical, semantic and profile identity"


def _probe_excision_policy_snapshot(source_generation_id: str) -> ExcisionPolicySnapshot:
    """Build one deterministic policy snapshot for a durable train probe."""
    from polylogue.security.excision_policy import ExcisionPolicySnapshot

    return ExcisionPolicySnapshot(
        removed_hashes=(bytes(range(32)),),
        assertion_refs=("assertion:durable-change-train-probe",),
        user_generation=3,
        audit_generation=4,
        audit_head="a" * 64,
        source_generation_id=source_generation_id,
    )


def _probe_source_generation_publish(publish: Callable[..., object]) -> str:
    """Exercise manifest-coordinate publication against the canonical source schema."""
    generation_id = "durable-change-train-source-generation"
    with _runtime_probe_source_connection() as probe:
        ids = publish(
            probe,
            source_generation_id=generation_id,
            manifest_digest="0" * 64,
            addressing_mode="path",
            coordinates=("probe/one.jsonl", "probe/two.jsonl"),
            observed_at_ms=1_780_000_000_000,
            policy_snapshot=_probe_excision_policy_snapshot(generation_id),
        )
        generation_row = probe.execute(
            "SELECT item_count FROM source_generations WHERE source_generation_id = ?",
            (generation_id,),
        ).fetchone()
        item_count = probe.execute(
            "SELECT COUNT(*) FROM source_items WHERE source_generation_id = ?",
            (generation_id,),
        ).fetchone()
        policy_rows = probe.execute(
            "SELECT COUNT(*) FROM excision_policy_projections WHERE source_generation_id = ?",
            (generation_id,),
        ).fetchone()
    if not isinstance(ids, tuple) or len(ids) != 2 or generation_row != (2,) or item_count != (2,):
        raise DurableChangeTrainError("source generation probe did not publish every manifest coordinate")
    if policy_rows != (1,):
        raise DurableChangeTrainError("source generation probe did not record its excision policy projection")
    return f"published probe source generation with {len(ids)} pending items"


def _probe_excision_policy_projection_read(read: Callable[..., object]) -> str:
    """Read back one policy binding the ordinary writer left in canonical DDL."""
    from polylogue.storage.sqlite.archive_tiers.source_items import publish_source_generation

    generation_id = "durable-change-train-excision-policy-generation"
    snapshot = _probe_excision_policy_snapshot(generation_id)
    with _runtime_probe_source_connection() as probe:
        absent = read(probe, generation_id)
        publish_source_generation(
            probe,
            source_generation_id=generation_id,
            manifest_digest="4" * 64,
            addressing_mode="path",
            coordinates=("probe/policy.jsonl",),
            observed_at_ms=1_780_000_000_000,
            policy_snapshot=snapshot,
        )
        projection = read(probe, generation_id)
    if absent is not None:
        raise DurableChangeTrainError("excision policy probe read a binding before one was published")
    if not isinstance(projection, dict):
        raise DurableChangeTrainError("excision policy probe did not read back a published binding")
    expected_digest = snapshot.digest
    if projection.get("policy_digest") != expected_digest or projection.get("audit_head") != "a" * 64:
        raise DurableChangeTrainError("excision policy probe read a binding that is not the one published")
    return "read back one generation-local excision policy binding from canonical DDL"


def _probe_source_attachment_record(record: Callable[..., object]) -> str:
    """Exercise the source attachment denominator's idempotent writer."""
    from polylogue.storage.sqlite.archive_tiers.source_attachments import SourceAttachment
    from polylogue.storage.sqlite.archive_tiers.source_items import publish_source_generation

    generation_id = "durable-change-train-attachment-generation"
    attachment = SourceAttachment(
        reference_id="attachment:durable-change-train",
        origin="codex-session",
        source_class="durable-change-train-probe",
        disposition="policy_rejected",
        reason="probe",
    )
    with _runtime_probe_source_connection() as probe:
        publish_source_generation(
            probe,
            source_generation_id=generation_id,
            manifest_digest="2" * 64,
            addressing_mode="path",
            coordinates=(),
            observed_at_ms=1_780_000_000_000,
        )
        record(
            probe,
            source_generation_id=generation_id,
            attachments=(attachment,),
            observed_at_ms=1_780_000_000_000,
        )
        record(
            probe,
            source_generation_id=generation_id,
            attachments=(attachment,),
            observed_at_ms=1_780_000_000_001,
        )
        row = probe.execute(
            "SELECT COUNT(*), disposition FROM source_attachments WHERE source_generation_id = ?",
            (generation_id,),
        ).fetchone()
    if row != (1, "policy_rejected"):
        raise DurableChangeTrainError("source attachment probe did not preserve idempotent publication")
    return "recorded one source attachment and replayed it idempotently"


def _probe_source_attachment_census(census: Callable[..., object]) -> str:
    """Exercise the source attachment census against a real canonical table."""
    from polylogue.storage.sqlite.archive_tiers.source_items import publish_source_generation

    generation_id = "durable-change-train-attachment-census-generation"
    with _runtime_probe_source_connection() as probe:
        publish_source_generation(
            probe,
            source_generation_id=generation_id,
            manifest_digest="3" * 64,
            addressing_mode="path",
            coordinates=(),
            observed_at_ms=1_780_000_000_000,
        )
        report = census(probe, generation_id)
    if not isinstance(report, dict) or report.get("source_generation_id") != generation_id:
        raise DurableChangeTrainError("source attachment census probe did not identify its generation")
    if report.get("pending") != 0 or report.get("sealable") is not True:
        raise DurableChangeTrainError("source attachment census probe reported unexpected pending work")
    return "read an empty source attachment denominator as sealable"


def _probe_source_generation_census(census: Callable[..., object]) -> str:
    """Exercise reconciliation census against the canonical source schema."""
    from polylogue.storage.sqlite.archive_tiers.source_items import publish_source_generation

    generation_id = "durable-change-train-census-generation"
    with _runtime_probe_source_connection() as probe:
        publish_source_generation(
            probe,
            source_generation_id=generation_id,
            manifest_digest="1" * 64,
            addressing_mode="path",
            coordinates=("probe/census.jsonl",),
            observed_at_ms=1_780_000_000_000,
        )
        report = census(probe, generation_id)
    if not isinstance(report, dict) or report.get("source_generation_id") != generation_id:
        raise DurableChangeTrainError("source generation census probe did not report the published generation")
    if report.get("sealable"):
        raise DurableChangeTrainError("source generation census probe reported a pending manifest as sealable")
    return "census reported the probe generation as pending and unsealable"


@contextmanager
def _runtime_probe_directory(*, prefix: str) -> Iterator[Path]:
    """Retain the original probe tree until every physical SQL owner retires."""
    from polylogue.core.sql_settlement import retain_native_sql_lifetimes
    from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_for_lifetime

    scratch = tempfile.TemporaryDirectory(prefix=prefix)
    with retain_native_sql_lifetimes(scratch):
        try:
            yield Path(scratch.name)
        finally:
            if not retained_native_sql_owners_for_lifetime(scratch):
                scratch.cleanup()
            # Otherwise original Native custody retains this exact directory;
            # final creator retirement releases its original cleanup finalizer.


def _probe_material_admission(admit: Callable[..., object]) -> str:
    """Exercise prepared claim admission against the canonical source schema."""
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.materials import prepare_material

    with _runtime_probe_directory(prefix="material-consumer-probe-") as directory:
        root = Path(directory)
        prepared = prepare_material(
            blob_store=ArchiveBlobPublisher(root / "source.db", root / "blob"),
            source_uri="https://durable-change-train.invalid/material-probe",
            referrer_ref="session:durable-change-train-probe",
        )
        with _runtime_probe_source_connection() as probe:
            observation = admit(
                probe,
                prepared=prepared,
                observed_at_ms=1_780_000_000_000,
            )
            material_id = getattr(observation, "material_id", None)
            row = probe.execute(
                "SELECT acquisition_state FROM material_observations WHERE material_id = ?",
                (material_id,),
            ).fetchone()
    if material_id is None or row is None or row[0] != "claimed":
        raise DurableChangeTrainError("material admission probe did not persist a claimed observation")
    return f"admitted probe material {str(material_id)[:12]} as a claimed observation"


def _probe_material_read(get: Callable[..., object]) -> str:
    """Exercise material read-back against the canonical source schema."""
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.materials import admit_material, prepare_material

    with _runtime_probe_directory(prefix="material-consumer-probe-") as directory:
        root = Path(directory)
        prepared = prepare_material(
            blob_store=ArchiveBlobPublisher(root / "source.db", root / "blob"),
            source_uri="https://durable-change-train.invalid/material-read-probe",
            referrer_ref="session:durable-change-train-read-probe",
        )
        with _runtime_probe_source_connection() as probe:
            observation = admit_material(
                probe,
                prepared=prepared,
                observed_at_ms=1_780_000_000_000,
            )
            loaded = get(probe, observation.material_id)
    if loaded is None or getattr(loaded, "source_uri", None) != observation.source_uri:
        raise DurableChangeTrainError("material read probe did not return the admitted observation")
    return f"read back probe material {observation.material_id[:12]}"


@contextmanager
def _runtime_probe_user_connection() -> Iterator[sqlite3.Connection]:
    """Own the actual canonical user-tier probe through native settlement."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_runtime_tier_probe

    with sqlite_connection(":memory:") as connection:
        initialize_runtime_tier_probe(connection, ArchiveTier.USER)
        yield connection


def _probe_query_promotion(promote: Callable[..., object]) -> str:
    """Exercise privacy-contracted promotion against the canonical user schema."""
    from polylogue.storage.sqlite.query_objects import get_query, put_query

    with _runtime_probe_user_connection() as probe:
        query = put_query(
            probe,
            {"field": "body", "value": "durable-change-train promotion probe"},
            grain="session",
            lane="dialogue",
            rank_policy="mixed",
            created_at_ms=1_780_000_000_000,
        )
        promote(
            probe,
            query_hash=query.query_hash,
            privacy_class="secret",
            retention_policy={"days": 1},
            excision_link="excision:durable-change-train-probe",
            promoted_at_ms=1_780_000_000_001,
        )
        promoted = get_query(probe, query.query_hash)
    if (
        promoted is None
        or promoted.privacy_class != "secret"
        or promoted.retention_policy != {"days": 1}
        or promoted.excision_link != "excision:durable-change-train-probe"
    ):
        raise DurableChangeTrainError("query promotion probe did not persist the privacy contract")
    return f"promoted probe query {query.query_hash[:12]} under a complete privacy contract"


def _probe_query_excision(apply_excision: Callable[..., object]) -> str:
    """Exercise excise-and-tombstone against the canonical user schema."""
    from polylogue.security.query_excision import plan_query_excision
    from polylogue.storage.sqlite.query_objects import get_query, promote_query, put_query

    with _runtime_probe_user_connection() as probe:
        query = put_query(
            probe,
            {"field": "body", "value": "durable-change-train excision probe"},
            grain="session",
            lane="dialogue",
            rank_policy="mixed",
            created_at_ms=1_780_000_000_000,
        )
        promote_query(
            probe,
            query_hash=query.query_hash,
            privacy_class="secret",
            retention_policy={"days": 1},
            excision_link="excision:durable-change-train-probe",
            promoted_at_ms=1_780_000_000_001,
        )
        plan = plan_query_excision(probe, query.ref)
        receipt = apply_excision(
            probe,
            plan,
            reason="durable-change-train probe",
            actor="agent:durable-change-train",
            now_ms=1_780_000_000_002,
        )
        remaining = get_query(probe, query.query_hash)
        ledger_row = probe.execute(
            "SELECT ledger_id FROM query_excision_ledger WHERE query_hash = ?",
            (query.query_hash,),
        ).fetchone()
    if getattr(receipt, "status", None) != "applied" or remaining is not None or ledger_row is None:
        raise DurableChangeTrainError("query excision probe did not excise the promoted query into the ledger")
    return f"excised probe query {query.query_hash[:12]} with a non-resurrection ledger row"


def _probe_assertion_upsert(upsert: Callable[..., object]) -> str:
    """Exercise the assertion writer against the canonical user schema.

    The rider's behavior proof is ``upsert-resolves-absent-status``: an
    ordinary write that supplies no status must land a resolved status value,
    never a NULL. Slot 002 makes ``assertions.status`` NOT NULL, so a writer
    that ever passed the caller's ``None`` straight through would fail here
    instead of silently relying on the column's nullability.
    """
    assertion_id = "durable-change-train-assertion"
    with _runtime_probe_user_connection() as probe:
        envelope = upsert(
            probe,
            assertion_id=assertion_id,
            target_ref="session:durable-change-train",
            kind="note",
            body_text="durable-change-train assertion probe",
            author_ref="user:durable-change-train",
            author_kind="user",
            now_ms=1_780_000_000_000,
        )
        stored = probe.execute(
            "SELECT status FROM assertions WHERE assertion_id = ?",
            (assertion_id,),
        ).fetchone()
    if stored is None or stored[0] is None:
        raise DurableChangeTrainError("assertion upsert probe left no resolved status on the durable row")
    returned_status = getattr(envelope, "status", None)
    if returned_status is None or str(returned_status.value) != str(stored[0]):
        raise DurableChangeTrainError("assertion upsert probe returned a status its durable row does not carry")
    return f"upserted probe assertion with resolved status {stored[0]!r} and no NULL fallback"


def _probe_assertion_status_mark(mark: Callable[..., object]) -> str:
    """Exercise the status marker against the canonical user schema.

    The rider's behavior proof is ``mark-needs-no-null-coalesce``. Fresh user
    DDL declares the status column non-null, so the marker's legacy
    ``COALESCE(status, 'active')`` guard is redundant. This probe pins that the
    marker still moves a row to a new terminal status and refuses a no-op
    restatement of the current one.
    """
    from polylogue.core.enums import AssertionStatus
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

    assertion_id = "durable-change-train-assertion-mark"
    with _runtime_probe_user_connection() as probe:
        upsert_assertion(
            probe,
            assertion_id=assertion_id,
            target_ref="session:durable-change-train",
            kind="note",
            body_text="durable-change-train status probe",
            author_ref="user:durable-change-train",
            author_kind="user",
            now_ms=1_780_000_000_000,
        )
        moved = mark(probe, assertion_id, AssertionStatus.SUPERSEDED, now_ms=1_780_000_000_001)
        restated = mark(probe, assertion_id, AssertionStatus.SUPERSEDED, now_ms=1_780_000_000_002)
        stored = probe.execute(
            "SELECT status FROM assertions WHERE assertion_id = ?",
            (assertion_id,),
        ).fetchone()
    if moved is not True or restated is not False:
        raise DurableChangeTrainError("assertion status probe did not move exactly one row to a new status")
    if stored is None or stored[0] != AssertionStatus.SUPERSEDED.value:
        raise DurableChangeTrainError("assertion status probe did not persist the marked status")
    return "marked one probe assertion superseded and refused the restatement"


@contextmanager
def _runtime_probe_user_file_connection() -> Iterator[sqlite3.Connection]:
    """Keep the real file probe and directory until native SQL settles."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_runtime_tier_probe
    from polylogue.storage.sqlite.connection_profile import scratch_connection_context

    with scratch_connection_context(prefix="polylogue-user-cursor-probe-", filename="user.db") as connection:
        cursor = connection.execute("PRAGMA database_list")
        try:
            probe_path = next(Path(str(row[2])) for row in cursor if row[1] == "main")
        finally:
            cursor.close()
        initialize_runtime_tier_probe(connection, ArchiveTier.USER, probe_path=probe_path)
        yield connection


def _probe_session_marker_delivery_writer(writer: Callable[..., object]) -> str:
    """Exercise the v003 per-session cursor through its real user writer."""
    with _runtime_probe_user_file_connection() as probe:
        writer(probe, session_id="session:marker-probe", input_binding="first", applied_at_ms=10)
        writer(probe, session_id="session:marker-probe", input_binding="older", applied_at_ms=9)
        writer(probe, session_id="session:marker-probe", input_binding="newer", applied_at_ms=11)
        stored = probe.execute(
            "SELECT input_binding, applied_at_ms FROM session_marker_delivery WHERE session_id = ?",
            ("session:marker-probe",),
        ).fetchone()
    if stored != ("newer", 11):
        raise DurableChangeTrainError("session marker delivery writer did not retain its monotonic durable position")
    return "advanced the v003 session marker cursor monotonically in a file-backed user tier"


def _probe_accepted_marker_delivery_cursor_writer(writer: Callable[..., object]) -> str:
    """Exercise the v004 source-stream cursor through its canonical writer."""
    from polylogue.storage.sqlite.archive_tiers.user_write import accepted_marker_delivery_cursor

    with _runtime_probe_user_file_connection() as probe:
        writer(
            probe,
            stream_id="durable-marker-stream",
            applied_sequence=1,
            applied_at_ms=10,
            expected_prior_sequence=0,
        )
        try:
            writer(
                probe,
                stream_id="durable-marker-stream",
                applied_sequence=1,
                applied_at_ms=11,
                expected_prior_sequence=1,
            )
        except ValueError:
            pass
        else:
            raise DurableChangeTrainError("accepted marker cursor writer accepted a duplicate source sequence")
        writer(
            probe,
            stream_id="durable-marker-stream",
            applied_sequence=2,
            applied_at_ms=12,
            expected_prior_sequence=1,
        )
        stored = accepted_marker_delivery_cursor(probe)
    if stored != ("durable-marker-stream", 2):
        raise DurableChangeTrainError("accepted marker cursor writer did not retain the contiguous source position")
    return "advanced one contiguous accepted-marker source cursor in a file-backed user tier"


def _probe_accepted_marker_delivery_cursor_reader(reader: Callable[..., object]) -> str:
    """Exercise the v004 cursor reader against a row written by the real writer."""
    from polylogue.storage.sqlite.archive_tiers.user_write import advance_accepted_marker_delivery_cursor

    with _runtime_probe_user_file_connection() as probe:
        advance_accepted_marker_delivery_cursor(
            probe,
            stream_id="durable-marker-stream",
            applied_sequence=1,
            applied_at_ms=10,
            expected_prior_sequence=0,
        )
        stored = reader(probe)
    if stored != ("durable-marker-stream", 1):
        raise DurableChangeTrainError("accepted marker cursor reader did not return the writer's durable position")
    return "read back the accepted-marker source cursor from a file-backed user tier"


def _probe_raw_state_update_compile(compiler: Callable[..., object]) -> str:
    """Exercise source v33's ``detected_provider`` rider on the write path.

    The rider's behavior proof is ``preserve-acquisition-origin``: recording a
    *detected* provider must never restate the acquisition origin, and a state
    update that resolves no provider must leave any already-detected value
    standing rather than nulling it. Both are properties of the compiled SQL,
    so the probe compiles rather than writes and needs no archive.
    """
    from polylogue.core.enums import Provider
    from polylogue.storage.raw.models import RawSessionStateUpdate

    resolved = compiler(RawSessionStateUpdate(payload_provider=Provider.CODEX), now_ms=0)
    unresolved = compiler(RawSessionStateUpdate(payload_provider=None), now_ms=0)
    if not isinstance(resolved, tuple) or not isinstance(unresolved, tuple):
        raise DurableChangeTrainError("raw state update probe did not compile to a (clauses, params) pair")
    resolved_clauses, resolved_params = cast(tuple[tuple[str, ...], tuple[object, ...]], resolved)
    unresolved_clauses, _ = cast(tuple[tuple[str, ...], tuple[object, ...]], unresolved)
    for clauses in (resolved_clauses, unresolved_clauses):
        if any(clause.split("=", 1)[0].strip() == "origin" for clause in clauses):
            raise DurableChangeTrainError("raw state update compiles a write to the immutable acquisition origin")
    if not any("detected_provider = COALESCE(" in clause for clause in unresolved_clauses):
        raise DurableChangeTrainError("raw state update would clobber detected_provider when no provider is resolved")
    if Provider.CODEX.value not in resolved_params:
        raise DurableChangeTrainError("raw state update did not bind the resolved detected provider")
    return f"compiled detected_provider rider: {len(resolved_clauses)} clause(s), acquisition origin untouched"


def _seed_probe_raw_row(
    connection: sqlite3.Connection,
    *,
    raw_id: str,
    source_path: str,
    blob_hash: bytes,
    source_index: int = 0,
    blob_size: int = 0,
    origin: str = "claude-code-session",
    acquired_at_ms: int = 1_780_000_000_000,
    detected_provider: str | None = None,
    parse_error: str | None = None,
) -> None:
    """Seed one synthetic ``raw_sessions`` row inside a throwaway probe archive.

    polylogue-1fijp: this is the only ``INSERT INTO raw_sessions`` left in the
    tree outside ``source_write.py``'s writer primitives, and a census should
    be able to tell at a glance that it is NOT an acquisition path. Every
    caller is a ``_probe_*`` function operating on an isolated canonical
    probe or archive; nothing here ever touches a live operator archive.

    These rows also cannot go through
    :func:`~polylogue.storage.sqlite.archive_tiers.raw_admission.admit_raw_observation`,
    and the reason is the point of the probes: each seeds a DELIBERATELY
    unusual row -- a ``detected_provider`` that disagrees with ``origin``, a
    row carrying a ``parse_error``, a bare row with no revision envelope --
    precisely to prove a migrated reader still handles that shape. That
    writer would normalize the row into a well-formed envelope and destroy
    the condition under test.
    """
    columns = ["raw_id", "origin", "source_path", "source_index", "blob_hash", "blob_size", "acquired_at_ms"]
    values: list[object] = [raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms]
    if detected_provider is not None:
        columns.append("detected_provider")
        values.append(detected_provider)
    if parse_error is not None:
        columns.append("parse_error")
        values.append(parse_error)
    connection.execute(
        f"INSERT INTO raw_sessions ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
        values,
    )


def _probe_captured_zip_admission(execute: Callable[..., object]) -> str:
    """Publish the captured coordinate atomically through actual raw admission."""
    from dataclasses import replace

    from polylogue.core.enums import Origin, Provider
    from polylogue.core.raw_coordinates import (
        CapturedZipMemberCoordinate,
        MemberAddressingMode,
        captured_zip_member_raw_id,
    )
    from polylogue.storage.sqlite.archive_tiers.raw_admission import (
        PendingPreParseRawAdmissionRequest,
        RawAdmissionResult,
        plan_raw_admission,
    )
    from polylogue.storage.sqlite.archive_tiers.source_write import read_raw_captured_zip_coordinate

    coordinate = CapturedZipMemberCoordinate(
        "/durable-change-train/physical/input.zip",
        "/durable-change-train/declared/input.zip",
        "directory:with:separators/session.json",
        2,
        0,
        MemberAddressingMode.WHOLE_MEMBER,
        "a" * 64,
        "c" * 64,
    )
    payload = b'{"probe":"captured-member"}'
    blob_hash = hashlib.sha256(payload).digest()
    raw_id = captured_zip_member_raw_id(coordinate, blob_hash.hex())
    request = PendingPreParseRawAdmissionRequest(
        origin=Origin.UNKNOWN_EXPORT,
        capture_mode=Provider.UNKNOWN,
        source_path=coordinate.declared_member,
        canonical_source_path=coordinate.canonical_member,
        source_index=coordinate.source_index,
        blob_hash=blob_hash,
        blob_size=len(payload),
        acquired_at_ms=1,
        raw_id=raw_id,
        captured_zip_coordinate=coordinate,
        addressing_mode=coordinate.addressing_mode.value,
        content_identity=blob_hash.hex(),
    )
    with _runtime_probe_source_connection() as probe:
        result = cast(RawAdmissionResult, execute(probe, plan_raw_admission(request)))
        if result.raw_id != raw_id or read_raw_captured_zip_coordinate(probe, raw_id) != coordinate:
            raise DurableChangeTrainError("captured ZIP admission lost its exact coordinate")
        before = probe.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]
        for altered in (
            replace(request, raw_id="unrelated-raw"),
            replace(request, source_path="/durable-change-train/other.zip:session.json"),
            replace(request, canonical_source_path=coordinate.declared_member),
            replace(request, source_index=coordinate.source_index + 1),
            replace(request, addressing_mode=MemberAddressingMode.ELEMENT_OF_CONTAINER.value),
        ):
            try:
                execute(probe, plan_raw_admission(altered))
            except ValueError:
                pass
            else:
                raise DurableChangeTrainError("captured ZIP admission accepted conflicting input evidence")
            if probe.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] != before:
                raise DurableChangeTrainError("captured ZIP refusal published an unrelated raw")
    return "admitted the captured coordinate and refused altered identity, paths, index and reading"


def _probe_captured_zip_restoration(resolve: Callable[..., object]) -> str:
    """Restore the captured physical member without reinterpreting path separators."""
    from polylogue.core.raw_coordinates import (
        CapturedZipMemberCoordinate,
        MemberAddressingMode,
        captured_zip_coordinate_receipt,
    )

    coordinate = CapturedZipMemberCoordinate(
        "/durable-change-train/physical/input.zip",
        "/durable-change-train/declared/input.zip",
        "directory:with:separators/session.json",
        2,
        0,
        MemberAddressingMode.WHOLE_MEMBER,
        "a" * 64,
        "c" * 64,
    )
    row = {
        "source_path": coordinate.declared_member,
        "captured_coordinate": captured_zip_coordinate_receipt(coordinate),
    }
    if resolve(row, Path("/durable-change-train/active")) != (coordinate.canonical_member, True):
        raise DurableChangeTrainError("captured ZIP restoration lost the accepted physical member")
    for receipt in (17, "{}"):
        try:
            resolve({**row, "captured_coordinate": receipt}, Path("/durable-change-train/active"))
        except ValueError:
            pass
        else:
            raise DurableChangeTrainError("captured ZIP restoration reinterpreted malformed evidence")
    return "restored the exact captured physical member and refused malformed receipts"


def _probe_raw_byte_revision_dependents(reader: Callable[..., object]) -> str:
    """Exercise both indexed dependency arms and the excluded self row."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_runtime_tier_probe

    with sqlite_connection(":memory:") as connection:
        initialize_runtime_tier_probe(connection, ArchiveTier.SOURCE)
        for raw_id in ("parent", "child"):
            _seed_probe_raw_row(connection, raw_id=raw_id, source_path=f"/{raw_id}.jsonl", blob_hash=b"0" * 32)
        connection.execute(
            "UPDATE raw_sessions SET predecessor_raw_id = 'parent', baseline_raw_id = 'parent' WHERE raw_id = 'parent'"
        )
        if reader(connection, "parent") or reader(connection, "missing"):
            raise DurableChangeTrainError("dependency reader counted its own row or an absent dependency")
        for column in ("predecessor_raw_id", "baseline_raw_id"):
            connection.execute(f"UPDATE raw_sessions SET {column} = 'parent' WHERE raw_id = 'child'")
            if not reader(connection, "parent"):
                raise DurableChangeTrainError(f"dependency reader missed {column}")
            connection.execute(f"UPDATE raw_sessions SET {column} = NULL WHERE raw_id = 'child'")
        if reader(connection, "parent"):
            raise DurableChangeTrainError("dependency reader retained a removed dependency")
    return "read predecessor and baseline dependents while excluding self and absent dependencies"


def _probe_revision_provider_resolution(descriptor: Callable[..., object]) -> str:
    """Prove replay routes on the detected provider, not the acquisition origin.

    This is the read side of the same rider: when a parser has recorded which
    provider it actually recognised, revision replay must route on that rather
    than re-deriving a provider from the origin the bytes were acquired under.
    The probe seeds a raw whose two values disagree so a descriptor that fell
    back to the origin would resolve the wrong parser.
    """
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    with _runtime_probe_directory(prefix="polylogue-durable-train-revision-") as directory:
        root = Path(directory) / "archive"
        initialize_active_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CLAUDE_CODE,
                payload=b'{"durable-change-train": "revision-probe"}\n',
                source_path="/durable-change-train/revision-probe.jsonl",
                canonical_source_path="/durable-change-train/revision-probe.jsonl",
                acquired_at_ms=1_780_000_000_000,
            )
            archive._ensure_source_conn().execute(
                "UPDATE raw_sessions SET detected_provider = ?, revision_kind = 'full' WHERE raw_id = ?",
                (Provider.CODEX.value, raw_id),
            )
            resolved = descriptor(archive, raw_id)
        if not isinstance(resolved, tuple) or not resolved:
            raise DurableChangeTrainError("revision provider resolution did not return a descriptor tuple")
        provider = resolved[0]
        if getattr(provider, "value", provider) != Provider.CODEX.value:
            raise DurableChangeTrainError(
                f"revision provider resolution ignored the detected provider and routed on origin: {provider!r}"
            )
    return "resolved a revision descriptor on the detected provider over the acquisition origin"


def _probe_raw_record_hydration(mapper: Callable[..., object]) -> str:
    """Prove hydration keeps v33's detected provider distinct from the origin.

    The rider's point is that a parser may record the provider it actually
    recognised without rewriting the immutable acquisition origin. A mapper
    that collapsed the two would satisfy every column-level check while
    silently destroying that distinction on read, so the probe hydrates a row
    whose two values deliberately disagree.
    """
    from polylogue.core.enums import Origin, Provider
    from polylogue.core.sources import provider_from_origin

    with _runtime_probe_source_connection() as connection:
        connection.row_factory = sqlite3.Row
        _seed_probe_raw_row(
            connection,
            raw_id="durable-change-train-hydration-raw",
            source_path="/durable-change-train/hydration-probe.jsonl",
            blob_hash=b"\0" * 32,
            detected_provider="codex",
        )
        row = connection.execute(
            "SELECT * FROM raw_sessions WHERE raw_id = ?",
            ("durable-change-train-hydration-raw",),
        ).fetchone()
        if row is None:
            raise DurableChangeTrainError("raw record hydration probe could not read back its seeded row")
        record = mapper(row)
        # Hydration splits the two: ``source_name`` carries the provider derived
        # from the immutable acquisition origin, ``payload_provider`` carries
        # what the parser actually detected. Collapsing them is the regression
        # this rider exists to prevent.
        acquisition = provider_from_origin(Origin.CLAUDE_CODE_SESSION)
        source_name = getattr(record, "source_name", None)
        detected = getattr(record, "payload_provider", None)
        if source_name != acquisition.value:
            raise DurableChangeTrainError(
                f"raw record hydration rewrote the acquisition origin: {source_name!r} (expected {acquisition.value!r})"
            )
        if getattr(detected, "value", detected) != Provider.CODEX.value:
            raise DurableChangeTrainError(f"raw record hydration lost the detected provider: {detected!r}")
    return "hydrated a raw record preserving both acquisition origin and detected provider"


def _probe_raw_failure_lifecycle(reader: Callable[..., object], archive_root: Path) -> str:
    """Exercise the source-tier failure lifecycle reader against the canonical source schema."""
    del archive_root
    # The numbered train authenticates this exact schema, which may precede
    # the runtime floor. Ordinary file admission must still reject that skew.
    with _runtime_probe_source_connection() as connection:
        connection.execute("PRAGMA query_only = ON")
        with connection:
            connection.execute("BEGIN")
            snapshot = reader(None, sample_limit=1, _connection=connection)
    if not getattr(snapshot, "available", False):
        raise DurableChangeTrainError(
            f"raw failure lifecycle probe could not read its snapshot: {getattr(snapshot, 'reason', None)}"
        )
    return f"read raw failure lifecycle state={getattr(snapshot, 'state', 'unknown')}"


@contextmanager
def _open_existing_tier(tier_path: Path) -> Iterator[sqlite3.Connection]:
    """Open an existing durable tier without allowing SQLite to create it.

    The connection is owned for the block: committed or rolled back like the
    builtin ``with sqlite3.connect(...)`` form, and always closed. Startup
    reconciliation runs on every archive open, so a connection left to the
    collector here retains three descriptors per open.
    """
    from polylogue.storage.sqlite.population_admission import assert_population_admitted
    from polylogue.storage.sqlite.write_lease import require_write_lease

    assert_population_admitted(tier_path)
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
    require_write_lease("open existing durable tier", archive_root=tier_path.parent)
    opened = False
    try:
        with sqlite_connection(f"{tier_path.resolve(strict=True).as_uri()}?mode=rw", uri=True) as connection:
            opened = True
            yield connection
    except (OSError, sqlite3.Error) as exc:
        if opened:
            raise
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
        _assert_durable_database_continuity(actual, expected, label=train.tier.value)
    except DurableChangeTrainError as exc:
        raise DurableChangeTrainError(
            f"{train.tier.value} durable tier continuity proof failed; refusing startup initialization/release"
        ) from exc


def _historical_schema_evidence(train: DurableChangeTrain) -> DurableMigrationReplayProof:
    """Return the immutable historical schema proof a forward admission needs."""
    if train.apply_evidence is None or train.proof is None or train.schema_replay_proof is None:
        raise DurableChangeTrainError("released train lacks historical schema evidence for forward-version admission")
    historical = train.proof.schema_replay_proof
    validate_durable_migration_replay_proof(historical, recompute_installed_bindings=True)
    if not historical.from_version < train.target_version <= historical.target_version:
        raise DurableChangeTrainError(
            "released train lacks exact historical schema evidence for forward-version admission"
        )
    step = _migration_runner._durable_migration_replay_step(historical, train.target_version)
    if (
        historical.tier is not train.tier
        or historical.from_version > train.current_version
        or historical.target_version != train.schema_replay_proof.target_version
        or not historical.matches
        or step.version != train.target_version
        or train.pre_apply_evidence is None
        or step.before_schema_inventory_sha256 != train.pre_apply_evidence.schema_inventory_sha256
        or step.after_schema_inventory_sha256 != train.apply_evidence.post.schema_inventory_sha256
        or historical.chain_sha256 != train.schema_replay_proof.chain_sha256
    ):
        raise DurableChangeTrainError(
            "released train lacks exact historical schema evidence for forward-version admission"
        )
    return historical


def _released_live_schema_inventory_sha256(
    tier: ArchiveTier,
    live_version: int,
    manifests_by_target: dict[int, DurableChangeTrain],
) -> str:
    """Return the persisted replay witness for the exact released live version."""
    train = manifests_by_target.get(live_version)
    if train is None or train.state is not DurableChangeTrainState.RELEASED:
        raise DurableChangeTrainError(
            f"{tier.value} durable tier v{live_version} lacks a released train to witness its live schema"
        )
    historical = _historical_schema_evidence(train)
    if historical.tier is not tier or not historical.from_version < live_version <= historical.target_version:
        raise DurableChangeTrainError(
            f"{tier.value} durable tier v{live_version} has an unrelated schema replay witness"
        )
    step = _migration_runner._durable_migration_replay_step(historical, live_version)
    return step.after_schema_inventory_sha256


def _canonical_schema_inventory(tier: ArchiveTier, target_version: int) -> _migration_runner.DurableSchemaInventory:
    """Construct one declared version from its immutable baseline and steps."""
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_BASELINE_DDL_BY_TIER

    if type(target_version) is not int:
        raise DurableChangeTrainError("canonical schema inventory target version must be an integer")
    floor = DURABLE_MIGRATION_ADOPTION_FLOORS[tier]
    if not floor <= target_version <= _runtime_durable_version(tier):
        raise DurableChangeTrainError("canonical schema inventory target is not supported")
    archive_ddl = ARCHIVE_BASELINE_DDL_BY_TIER.get(tier)
    if not isinstance(archive_ddl, str):
        raise DurableChangeTrainError(f"no canonical archive baseline DDL is registered for {tier.value}")
    steps = tuple(step for step in _migration_runner._load_migrations(tier) if floor < step.version <= target_version)
    if tuple(step.version for step in steps) != tuple(range(floor + 1, target_version + 1)):
        raise DurableChangeTrainError("canonical schema inventory lacks its declared migration chain")
    return _canonical_schema_inventory_for_ddl(tier, target_version, archive_ddl, steps)


@lru_cache(maxsize=64)
def _canonical_schema_inventory_for_ddl(
    tier: ArchiveTier,
    target_version: int,
    archive_ddl: str,
    steps: tuple[_migration_runner.MigrationStep, ...],
) -> _migration_runner.DurableSchemaInventory:
    """Memoize only exact baseline/ordered installed SQL schema computation.

    Discovery validates the installed sidecar claims before every memo lookup.
    Each destination's physical application and live proofs remain uncached.
    The baseline and complete ordered steps are keys, so neither a different
    baseline nor changed installed SQL can reuse a prior schema inventory.
    """
    with sqlite_connection(":memory:") as fresh:
        fresh.execute("PRAGMA foreign_keys = ON")
        fresh.executescript(archive_ddl)
        fresh.execute(f"PRAGMA user_version = {DURABLE_MIGRATION_ADOPTION_FLOORS[tier]}")
        fresh.commit()
        for step in steps:
            with fresh:
                # The connection context settles a transaction; DDL does not start one.
                fresh.execute("BEGIN IMMEDIATE")
                _migration_runner._execute_proved_migration_sql(fresh, step)
                fresh.execute(f"PRAGMA user_version = {step.version}")
        if int(fresh.execute("PRAGMA user_version").fetchone()[0]) != target_version:
            raise DurableChangeTrainError("canonical schema inventory did not reach its declared target")
        return _migration_runner.capture_durable_schema_inventory(fresh)


def _verify_released_train_live_tier(
    conn: sqlite3.Connection,
    train: DurableChangeTrain,
    *,
    current_target_version: int | None = None,
    actual_evidence: _ReleasedSchemaEvidence | None = None,
    integrity_check: tuple[str, ...] | None = None,
    live_inventory: _migration_runner.DurableSchemaInventory | None = None,
    expected_live_schema_inventory_sha256: str | None = None,
) -> DurableForwardVersionReceipt | None:
    """Verify a released train remains represented after later trains advance it."""
    if train.apply_evidence is None:
        raise DurableChangeTrainError(f"{train.state.value} train lacks post-apply continuity evidence")
    actual = actual_evidence or _capture_released_schema_evidence(conn, train.tier)
    if actual.user_version < train.target_version:
        raise DurableChangeTrainError(
            f"{train.tier.value} durable tier continuity proof failed: live version regressed below released train "
            "target; refusing startup initialization"
        )
    if train.state is not DurableChangeTrainState.RELEASED:
        raise DurableChangeTrainError("historical admission requires a released train")
    historical = _historical_schema_evidence(train)
    expected_identity = train.apply_evidence.post.archive_identity_digest
    if actual.archive_identity_digest != expected_identity:
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
    if actual.user_version == train.target_version:
        # A released migration proves immutable schema history, not a freeze
        # of the rows subsequently acquired under the ordinary writer owner.
        # APPLIED/PROVEN recovery retains the exact post-content proof above.
        expected_live_schema_inventory_sha256 = _migration_runner._durable_migration_replay_step(
            historical, train.target_version
        ).after_schema_inventory_sha256
    elif expected_live_schema_inventory_sha256 is None:
        raise DurableChangeTrainError(
            f"{train.tier.value} durable tier v{actual.user_version} lacks persisted replay evidence for its live schema"
        )
    if live_inventory.sha256 != expected_live_schema_inventory_sha256:
        raise DurableChangeTrainError(
            f"{train.tier.value} durable tier schema inventory differs from the released migration replay witness "
            f"for live v{actual.user_version}"
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
    if actual.user_version == train.target_version:
        return None
    return DurableForwardVersionReceipt(
        tier=train.tier,
        historical_train_id=train.train_id,
        historical_target_version=train.target_version,
        current_target_version=runtime_target,
        observed_live_version=actual.user_version,
        historical_schema_inventory_sha256=_migration_runner._durable_migration_replay_step(
            historical, train.target_version
        ).after_schema_inventory_sha256,
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
            floor=_chain_floor(tier, _durable_chain_floor_versions(archive_root, manifest_root)),
        )
    if not historical:
        return None
    if evidence is None:
        actual = _capture_released_schema_evidence(conn, tier)
        evidence = _DurableForwardVersionEvidence(
            actual=actual,
            integrity_check=tuple(str(row[0]) for row in conn.execute("PRAGMA integrity_check")),
            live_inventory=_migration_runner.capture_durable_schema_inventory(conn),
            expected_live_schema_inventory_sha256=_released_live_schema_inventory_sha256(
                tier, actual.user_version, manifests_by_target
            ),
        )
    for train in sorted(historical, key=lambda item: item.target_version, reverse=True):
        receipt = _verify_released_train_live_tier(
            conn,
            train,
            current_target_version=current_target_version,
            actual_evidence=evidence.actual,
            integrity_check=evidence.integrity_check,
            live_inventory=evidence.live_inventory,
            expected_live_schema_inventory_sha256=evidence.expected_live_schema_inventory_sha256,
        )
        if receipt is not None:
            return receipt
    return None


def assert_released_durable_tier_lineage(archive_root: Path, tier: ArchiveTier, connection: sqlite3.Connection) -> None:
    """Prove a surviving advanced tier without mutating startup history.

    A birth marker proves its baseline shape. Above that version the actual
    physical file, installed migration chain and released schema witness
    must prove membership even when another durable tier is missing.
    """
    current = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if current > _runtime_durable_version(tier):
        raise DurableTierNewerThanRuntimeError(
            tier, live_version=current, runtime_version=_runtime_durable_version(tier)
        )
    if current <= DURABLE_MIGRATION_ADOPTION_FLOORS[tier]:
        return
    manifests = _released_train_manifests_by_target(archive_root / ".maintenance-state" / "durable-change-trains", tier)
    _require_released_train_chain(tier, manifests, current_version=current)
    _verify_released_train_live_tier(connection, manifests[current])


def _released_train_manifests_by_target(
    manifest_root: Path,
    tier: ArchiveTier,
) -> dict[int, DurableChangeTrain]:
    """Load one persisted train record per target version for a durable tier."""
    manifests_by_target: dict[int, DurableChangeTrain] = {}
    if not manifest_root.is_dir():
        return manifests_by_target
    for path in durable_train_manifest_paths(manifest_root, tier):
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


def _durable_chain_floor_versions(archive_root: Path, manifest_root: Path) -> dict[ArchiveTier, int]:
    """Return every version a durable tier reached without a numbered train.

    Only a fresh direct bootstrap puts a durable tier above its adoption floor
    with no train manifest to prove it. That is the archive's own evidence, so
    it raises the chain floor; trains are still required for every version
    above it.

    A released train's own pre-apply evidence is deliberately not a second
    route. It authenticates a marker the archive already carries
    (``_released_train_proven_floor``), but it never raises the floor on its
    own: a tier standing above the floor with released trains and no declared
    bootstrap evidence stays a refusal.
    """
    return dict(_fresh_durable_bootstrap_versions(archive_root, manifest_root))


def _retire_corroborated_fresh_durable_bootstrap_marker(
    manifest_root: Path,
    bootstrap_versions: dict[ArchiveTier, int],
) -> bool:
    """Delete the bootstrap marker once released trains carry its whole floor.

    The marker's only authority is the chain floor it grants. Once a tier has
    walked a complete released train chain from its adoption floor up to the
    version the marker records, that chain proves everything the marker did and
    the marker grants nothing -- so keeping it would gate the archive on
    bootstrap evidence for the rest of its life for no remaining benefit.

    Retirement is decided by running the ordinary chain requirement at the
    plain adoption floor: the marker is removed only when the archive still
    satisfies it with the marker's contribution taken away. A tier whose
    recorded version is already at or below its adoption floor grants nothing
    to begin with.
    """
    marker_path = manifest_root / _FRESH_DURABLE_BOOTSTRAP_MARKER
    if not marker_path.is_file() or not bootstrap_versions:
        return False
    for tier, version in bootstrap_versions.items():
        adoption_floor = DURABLE_MIGRATION_ADOPTION_FLOORS[tier]
        if version <= adoption_floor:
            continue
        try:
            _require_released_train_chain(
                tier,
                _released_train_manifests_by_target(manifest_root, tier),
                current_version=version,
                floor=adoption_floor,
            )
        except DurableChangeTrainError:
            return False
    marker_path.unlink(missing_ok=True)
    _migration_runner._fsync_manifest_directory(manifest_root)
    return True


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
        with _open_existing_tier(tier_path) as live:
            _verify_persisted_live_tier_continuity(live, train)
            if train.reservation is not None and train.reservation.active:
                previous_revision = train.revision
                train = record_durable_writer_release(
                    train,
                    evidence_ref=f"proof:startup-writer-release:{train.train_id}",
                )
                train = _persist_train_transition(manifest_path, train, expected_revision=previous_revision)
            actual_parity = train.schema_replay_proof
            if actual_parity is None:
                raise DurableChangeTrainError("applied train lost its persisted schema replay proof")
            validate_durable_migration_replay_proof(actual_parity, recompute_installed_bindings=True)
            replay_step = _migration_runner._durable_migration_replay_step(actual_parity, train.target_version)
            if replay_step.after_schema_inventory_sha256 != capture_durable_schema_inventory(live).sha256:
                raise DurableChangeTrainError("recovered live schema differs from the persisted replay step output")
            runtime_results = (
                tuple(runtime_consumer_results)
                if runtime_consumer_results is not None
                else _runtime_consumer_results(train, archive_root, candidate=live)
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
                schema_replay_proof=actual_parity,
                runtime_consumers=runtime_results,
                restart_convergence=restart,
                proof_refs=(f"proof:startup-recovery:{train.train_id}",),
            )
            _verify_persisted_live_tier_continuity(live, train)
            train = _persist_train_transition(manifest_path, train, expected_revision=previous_revision)
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
    schema_replay_proof: DurableMigrationReplayProof | None = None,
    release_archive_ownership: Callable[[], None],
) -> DurableChangeTrainExecution:
    """Execute every persisted train state while the caller holds archive ownership.

    The caller-held lease must cover startup reconciliation and receipt creation so
    reused forward-version evidence cannot become stale between those operations.
    """
    from polylogue.storage.sqlite.archive_tiers.archive_plan import assert_archive_format_lineage

    try:
        assert_archive_format_lineage(archive_root)
    except RuntimeError as exc:
        raise DurableChangeTrainError(str(exc)) from exc
    forward_version_evidence: dict[ArchiveTier, _DurableForwardVersionEvidence] = {}
    reconcile_durable_change_train_startup(archive_root, live_evidence_cache=forward_version_evidence)
    tier_path = archive_root / f"{tier.value}.db"
    with _open_existing_tier(tier_path) as probe:
        current_version = int(probe.execute("PRAGMA user_version").fetchone()[0] or 0)
        rehearsed_proof = schema_replay_proof
        if rehearsed_proof is None and current_version < _migration_runner.ARCHIVE_VERSION_BY_TIER[tier]:
            rehearsed_proof = rehearse_durable_migration_chain(
                probe,
                tier,
                target_version=_migration_runner.ARCHIVE_VERSION_BY_TIER[tier],
                evidence_ref=f"proof:schema-replay:{tier.value}:v{current_version}-to-current",
            )
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
        with sqlite_connection(":memory:") as preflight:
            _migration_runner._pending_migration_steps(
                preflight,
                tier,
                current_version=current_version,
                target_version=runtime_target_version,
            )
    legacy_result: MigrationResult | None = None
    floor = DURABLE_MIGRATION_ADOPTION_FLOORS.get(tier)
    if floor is not None and current_version < floor:
        with _open_existing_tier(tier_path) as conn:
            legacy_result = _migration_runner.migrate_archive_tier(
                conn,
                tier,
                backup_manifest=backup_manifest,
                target_version=floor,
            )
        current_version = legacy_result.to_version
    sidecar = durable_migration_sidecar_for_slot(tier, current_version + 1)
    if sidecar is not None and legacy_result is not None and legacy_result.applied_versions:
        # The legacy climb just rewrote the live tier, so the supplied manifest
        # now fingerprints bytes that no longer exist.  Backup authorization
        # deliberately binds the exact pre-apply bytes; reusing the pre-climb
        # manifest here would weaken train recovery to "restore and replay".
        # The climb is committed and safe — require a fresh backup for the train.
        raise DurableChangeTrainError(
            f"legacy migrations advanced {tier.value} to v{current_version}; the supplied backup manifest "
            f"covers the pre-migration tier and cannot authorize train v{sidecar.slot} "
            f"({sidecar.train.train_id}). Take a fresh verified backup of the migrated tier and run the "
            "train again."
        )
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
                expected_live_schema_inventory_sha256=(
                    forward_version_evidence[tier].expected_live_schema_inventory_sha256
                    if tier in forward_version_evidence
                    else None
                ),
            )
        return DurableChangeTrainExecution(
            train=train,
            manifest_path=manifest_path,
            migration_result=None,
            forward_version_receipt=forward_version_receipt,
        )

    if train.state is DurableChangeTrainState.DECLARED:
        if rehearsed_proof is None:
            raise DurableChangeTrainError(
                f"declared {tier.value} train {train.train_id} has no schema replay proof for admission"
            )
        previous_revision = train.revision
        train = admit_durable_change_train(
            train,
            observed_current_version=current_version,
            schema_replay_proof=rehearsed_proof,
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
        with _open_existing_tier(tier_path) as conn:
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
            with _open_existing_tier(tier_path) as conn:
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


def _runtime_durable_version(tier: ArchiveTier) -> int:
    return cast(dict[ArchiveTier, int], vars(_migration_runner)["ARCHIVE_VERSION_BY_TIER"])[tier]


def _refuse_durable_tiers_newer_than_runtime(archive_root: Path, *, recovering: set[ArchiveTier]) -> None:
    """Raise the typed refusal for a durable tier a newer release wrote."""
    for tier in DURABLE_MIGRATION_ADOPTION_FLOORS:
        tier_path = archive_root / f"{tier.value}.db"
        if tier in recovering or not tier_path.is_file():
            continue
        with _open_existing_tier(tier_path) as live:
            live_version = int(live.execute("PRAGMA user_version").fetchone()[0] or 0)
        if live_version > _runtime_durable_version(tier):
            raise DurableTierNewerThanRuntimeError(
                tier, live_version=live_version, runtime_version=_runtime_durable_version(tier)
            )


def _refuse_lost_durable_tiers(archive_root: Path, *, recovering: set[ArchiveTier]) -> None:
    """Refuse an established archive that lost a durable tier.

    The format marker is published only after all six tiers exist, so a
    marked root without one of its durable tiers lost it outside Polylogue.
    Nothing here may recreate it: an empty Source tier would silently claim
    an acquisition history the archive no longer has. A tier under an
    unreleased train is left to that train's recovery.
    """
    from polylogue.storage.sqlite.archive_tiers.archive_plan import archive_format_marker_path

    if not archive_format_marker_path(archive_root).is_file():
        return
    for tier in DURABLE_MIGRATION_ADOPTION_FLOORS:
        tier_path = archive_root / f"{tier.value}.db"
        if tier in recovering or tier_path.exists() or tier_path.is_symlink():
            continue
        raise DurableChangeTrainError(
            f"established archive is missing {tier_path.name}; a lost durable tier is never recreated. "
            "Restore the archive root from a verified backup"
        )


def _reconcile_durable_change_train_startup_locked(
    archive_root: Path,
    *,
    live_evidence_cache: dict[ArchiveTier, _DurableForwardVersionEvidence] | None = None,
) -> tuple[Path, ...]:
    """Reconcile persisted trains while the caller holds archive ownership."""
    manifest_root = archive_root / ".maintenance-state" / "durable-change-trains"
    reconciled: list[Path] = []
    live_evidence_by_tier: dict[ArchiveTier, _ReleasedSchemaEvidence] = {}
    live_integrity_by_tier: dict[ArchiveTier, tuple[str, ...]] = {}
    live_inventory_by_tier: dict[ArchiveTier, _migration_runner.DurableSchemaInventory] = {}
    expected_live_schema_inventory_by_tier: dict[ArchiveTier, str] = {}
    manifests_by_tier: dict[ArchiveTier, dict[int, DurableChangeTrain]] = {}
    validated_tiers: set[ArchiveTier] = set()
    manifest_paths = durable_train_manifest_paths(manifest_root)
    # Before the bootstrap marker is corroborated: a newer release's marker
    # records versions whose schema this runtime cannot reconstruct, so the
    # ownership proof would refuse it as foreign instead of naming the skew.
    # A tier with an unreleased train is left to the recovery below, whose
    # failure classification is the stronger refusal.
    recovering = {
        train.tier
        for train in map(load_durable_change_train_manifest, manifest_paths)
        if train.state is not DurableChangeTrainState.RELEASED
    }
    _refuse_durable_tiers_newer_than_runtime(archive_root, recovering=recovering)
    _refuse_lost_durable_tiers(archive_root, recovering=recovering)
    fresh_bootstrap_versions = _fresh_durable_bootstrap_versions(archive_root, manifest_root)
    _retire_corroborated_fresh_durable_bootstrap_marker(manifest_root, fresh_bootstrap_versions)
    chain_floor_versions = _durable_chain_floor_versions(archive_root, manifest_root)

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
        if train.schema_replay_proof is not None:
            validate_durable_migration_replay_proof(
                train.schema_replay_proof,
                recompute_installed_bindings=True,
            )
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
        runtime_version = _runtime_durable_version(tier)
        if current_version > runtime_version:
            # Only a newer release can have written this tier. No chain of
            # this runtime's trains can admit it, so say that plainly rather
            # than as missing train evidence (polylogue-w6nrl).
            raise DurableTierNewerThanRuntimeError(tier, live_version=current_version, runtime_version=runtime_version)
        if current_version <= adoption_floor:
            continue
        manifests_by_tier[tier] = _released_train_manifests_by_target(manifest_root, tier)
        tier_manifest_paths = durable_train_manifest_paths(manifest_root, tier)
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
            floor=_chain_floor(tier, chain_floor_versions),
        )
        validated_tiers.add(tier)

    for manifest_path in manifest_paths:
        train = load_durable_change_train_manifest(manifest_path)
        if train.schema_replay_proof is not None:
            validate_durable_migration_replay_proof(
                train.schema_replay_proof,
                recompute_installed_bindings=True,
            )
        if train.state is not DurableChangeTrainState.RELEASED:
            continue
        with _open_existing_tier(archive_root / f"{train.tier.value}.db") as live:
            actual = live_evidence_by_tier.get(train.tier)
            if actual is None:
                actual = _capture_released_schema_evidence(live, train.tier)
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
                    floor=_chain_floor(train.tier, chain_floor_versions),
                )
            if actual.user_version > train.target_version:
                if train.tier not in live_integrity_by_tier:
                    live_integrity_by_tier[train.tier] = tuple(
                        str(row[0]) for row in live.execute("PRAGMA integrity_check")
                    )
                if train.tier not in live_inventory_by_tier:
                    live_inventory_by_tier[train.tier] = _migration_runner.capture_durable_schema_inventory(live)
                if train.tier not in expected_live_schema_inventory_by_tier:
                    expected_live_schema_inventory_by_tier[train.tier] = _released_live_schema_inventory_sha256(
                        train.tier,
                        actual.user_version,
                        manifests_by_tier[train.tier],
                    )
                if live_evidence_cache is not None:
                    live_evidence_cache[train.tier] = _DurableForwardVersionEvidence(
                        actual=actual,
                        integrity_check=live_integrity_by_tier[train.tier],
                        live_inventory=live_inventory_by_tier[train.tier],
                        expected_live_schema_inventory_sha256=(expected_live_schema_inventory_by_tier[train.tier]),
                    )
            _verify_released_train_live_tier(
                live,
                train,
                actual_evidence=actual,
                integrity_check=live_integrity_by_tier.get(train.tier),
                live_inventory=live_inventory_by_tier.get(train.tier),
                expected_live_schema_inventory_sha256=expected_live_schema_inventory_by_tier.get(train.tier),
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
    "durable_train_manifest_paths",
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
    "DurableChangeTrainApplyError",
    "DurableChangeTrainRecoveryError",
    "DurableTierNewerThanRuntimeError",
    "DurableMigrationClaim",
    "durable_migration_claim_for_sql",
    "durable_migration_claims",
    "durable_migration_collision_report",
    "find_durable_migration_collisions",
    "add_durable_change_train_rider",
    "durable_change_train_from_payload",
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


def _probe_frontier_dependency_read(operation: Callable[..., object], *, journal: bool) -> str:
    """Exercise the installed journal and absent-measurement read contracts."""
    from contextlib import ExitStack

    from polylogue.storage.frontier_inspection import declared_frontier_triggers
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_runtime_tier_probe

    with tempfile.TemporaryDirectory(prefix="frontier-runtime-probe-") as directory, ExitStack() as stack:
        root = Path(directory)
        handles: dict[ArchiveTier, sqlite3.Connection] = {}
        for tier in (ArchiveTier.SOURCE, ArchiveTier.INDEX, ArchiveTier.OPS):
            path = root / f"{tier.value}.db"
            connection = stack.enter_context(sqlite_connection(path))
            initialize_runtime_tier_probe(connection, tier, probe_path=path)
            connection.commit()
            handles[tier] = connection
        if journal:
            result = operation(
                source=handles[ArchiveTier.SOURCE],
                index=handles[ArchiveTier.INDEX],
                ops=handles[ArchiveTier.OPS],
                source_path=root / "source.db",
                index_path=root / "index.db",
                ops_path=root / "ops.db",
                source_triggers=declared_frontier_triggers(ARCHIVE_DDL_BY_TIER[ArchiveTier.SOURCE]),
                index_triggers=declared_frontier_triggers(ARCHIVE_DDL_BY_TIER[ArchiveTier.INDEX]),
                cursor_triggers=declared_frontier_triggers(ARCHIVE_DDL_BY_TIER[ArchiveTier.OPS]),
            )
            if any(
                getattr(result, name) != 0
                for name in ("source_high", "source_floor", "index_high", "index_floor", "cursor_high", "cursor_floor")
            ):
                raise DurableChangeTrainError("empty installed frontier journal has unexpected coverage")
            return "read exact installed dependency triggers and empty three-tier journal coverage"
        if operation(handles[ArchiveTier.OPS]) is not None:
            raise DurableChangeTrainError("empty frontier measurement was falsely available")
        return "unmeasured installed frontier remains unavailable"

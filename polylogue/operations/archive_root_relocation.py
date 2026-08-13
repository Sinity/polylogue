"""One explicit offline transition for an inode-preserving archive-root move."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from polylogue.config import Config
from polylogue.maintenance.offline_guard import offline_writer_block_reason
from polylogue.maintenance.receipt_fs import (
    MaintenanceReceiptPathError,
    atomic_replace_receipt,
    existing_maintenance_receipt_directory,
    iter_pinned_receipts,
    maintenance_receipt_directory,
    read_optional_receipt,
)
from polylogue.paths import render_root
from polylogue.storage.archive_identity import (
    ArchiveIdentity,
    ArchiveLocation,
    ArchiveOwnershipError,
    OwnedArchiveLocation,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import (
    DURABLE_MIGRATION_ADOPTION_FLOORS,
    DurableChangeTrainError,
    DurableChangeTrainState,
    _released_train_manifests_by_target,
    _require_released_train_chain,
    _validate_source_continuity_refresh_receipt,
    load_durable_change_train_manifest,
    rebind_released_source_train_archive_identity,
    write_durable_change_train_manifest,
    write_source_continuity_relocation_transition,
)
from polylogue.storage.sqlite.migration_runner import (
    MigrationError,
    capture_durable_database_evidence,
    capture_durable_schema_inventory,
    validate_full_evidence_backup_for_archive_root_relocation,
)
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

PLAN_FORMAT: Literal["polylogue.archive-root-relocation-plan.v1"] = "polylogue.archive-root-relocation-plan.v1"
RECEIPT_FORMAT: Literal["polylogue.archive-root-relocation-receipt.v1"] = "polylogue.archive-root-relocation-receipt.v1"
_TIER_NAMES = tuple(tier.value for tier in ArchiveTier)
_DURABLE_TIER_NAMES = ("source", "user", "audit")
_SIDECARS = ("-wal", "-shm", "-journal")


class ArchiveRootRelocationError(RuntimeError):
    """The requested root move has no single safe offline transition."""


class RelocationTierEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: str
    configured_path: str
    resolved_path: str
    old_device: int
    old_inode: int
    device: int
    inode: int
    size_bytes: int
    sha256: str
    user_version: int
    schema_inventory_sha256: str
    content_sha256: str
    quick_check: tuple[str, ...]


class RelocationSourceTrain(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    before_revision: int
    before_manifest_sha256: str
    before_archive_identity_digest: str
    after_archive_identity_digest: str
    requires_rebind: bool
    source_continuity_receipt_digests: tuple[str, ...]


class ArchiveRootRelocationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["polylogue.archive-root-relocation-plan.v1"] = PLAN_FORMAT
    old_configured_root: str
    old_resolved_root: str
    old_root_device: int
    old_root_inode: int
    new_configured_root: str
    new_resolved_root: str
    new_root_device: int
    new_root_inode: int
    backup_manifest_path: str
    backup_manifest_sha256: str
    backup_receipt_path: str
    backup_receipt_sha256: str
    backup_profile: Literal["full_evidence"]
    backup_tier_inventory: tuple[str, ...]
    tiers: tuple[RelocationTierEvidence, ...]
    source_trains: tuple[RelocationSourceTrain, ...]
    stopped_daemon_evidence_ref: str
    single_writer_evidence_ref: str
    bound_confirmation: str
    plan_sha256: str


class ArchiveRootRelocationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["polylogue.archive-root-relocation-receipt.v1"] = RECEIPT_FORMAT
    state: Literal["prepared", "committed"]
    revision: int
    plan_sha256: str
    authorization: str
    manifest_before_sha256: tuple[str, ...]
    manifest_after_sha256: tuple[str, ...]
    resume_command: str
    prepared_receipt_sha256: str | None = None
    receipt_sha256: str


class ArchiveRootRelocationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: Literal[True] = True
    state: Literal["prepared", "committed"]
    plan_sha256: str
    receipt_path: str | None
    changed_manifests: tuple[str, ...]


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _sealed_plan(**values: object) -> ArchiveRootRelocationPlan:
    plan = ArchiveRootRelocationPlan.model_validate({"format": PLAN_FORMAT, **values, "plan_sha256": ""})
    payload = plan.model_dump(mode="json", exclude={"plan_sha256"})
    return plan.model_copy(update={"plan_sha256": _canonical_sha256(payload)})


def _sealed_receipt(**values: object) -> ArchiveRootRelocationReceipt:
    receipt = ArchiveRootRelocationReceipt.model_validate({"format": RECEIPT_FORMAT, **values, "receipt_sha256": ""})
    payload = receipt.model_dump(mode="json", exclude={"receipt_sha256"})
    return receipt.model_copy(update={"receipt_sha256": _canonical_sha256(payload)})


def _verify_plan(plan: ArchiveRootRelocationPlan) -> None:
    expected = _canonical_sha256(plan.model_dump(exclude={"plan_sha256"}, mode="json"))
    if plan.plan_sha256 != expected:
        raise ArchiveRootRelocationError("archive-root relocation plan checksum mismatch")


def _verify_receipt(receipt: ArchiveRootRelocationReceipt) -> None:
    expected = _canonical_sha256(receipt.model_dump(exclude={"receipt_sha256"}, mode="json"))
    if receipt.receipt_sha256 != expected:
        raise ArchiveRootRelocationError("archive-root relocation receipt checksum mismatch")


def _real_directory(path: Path, *, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ArchiveRootRelocationError(f"cannot inspect {label}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ArchiveRootRelocationError(f"{label} is not a real directory: {path}")
    absolute = Path(os.path.abspath(path))
    resolved = path.resolve(strict=True)
    if absolute != resolved:
        raise ArchiveRootRelocationError(f"{label} traverses a symbolic link: {path}")
    return resolved


def _real_file(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ArchiveRootRelocationError(f"cannot inspect {label}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ArchiveRootRelocationError(f"{label} is not a real single-linked file: {path}")
    return metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_sidecars(root: Path) -> None:
    for tier in _TIER_NAMES:
        for suffix in _SIDECARS:
            path = root / f"{tier}.db{suffix}"
            if path.exists() or path.is_symlink():
                raise ArchiveRootRelocationError(f"archive-root relocation refuses SQLite sidecar: {path}")


def _tier_snapshot(
    root: Path,
    tier: ArchiveTier,
    *,
    old_device: int,
    old_inode: int,
) -> RelocationTierEvidence:
    location = ArchiveLocation.resolve(root)
    identity = location.active_tier(tier.value)
    path = identity.configured_path
    resolved_path = identity.resolved_path
    if tier is ArchiveTier.INDEX:
        # The promoted index route deliberately uses an active-generation
        # pointer.  Snapshot the resolved generation, never a shadow path.
        metadata = _real_file(resolved_path, label="active index tier")
    else:
        metadata = _real_file(path, label=f"{tier.value} tier")
    try:
        with sqlite3.connect(f"file:{resolved_path}?mode=ro&immutable=1", uri=True) as connection:
            if tier is ArchiveTier.EMBEDDINGS:
                loaded, error = try_load_sqlite_vec(connection)
                if not loaded:
                    raise ArchiveRootRelocationError(
                        "cannot load sqlite-vec for immutable embeddings evidence"
                    ) from error
            schema = capture_durable_schema_inventory(connection)
            quick_check = tuple(str(row[0]) for row in connection.execute("PRAGMA quick_check"))
            user_version = int(connection.execute("PRAGMA user_version").fetchone()[0] or 0)
            if tier.value in _DURABLE_TIER_NAMES:
                evidence = capture_durable_database_evidence(connection, tier)
                content_sha256 = evidence.content_sha256
            else:
                content_sha256 = _sha256_file(resolved_path)
    except (OSError, sqlite3.Error, MigrationError) as exc:
        raise ArchiveRootRelocationError(f"cannot read {tier.value} tier without mutation") from exc
    return RelocationTierEvidence(
        tier=tier.value,
        configured_path=str(path.absolute()),
        resolved_path=str(resolved_path),
        old_device=old_device,
        old_inode=old_inode,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size_bytes=metadata.st_size,
        sha256=_sha256_file(resolved_path),
        user_version=user_version,
        schema_inventory_sha256=schema.sha256,
        content_sha256=content_sha256,
        quick_check=quick_check,
    )


def _source_trains(
    root: Path,
    *,
    source_version: int,
    source_content_sha256: str,
    after_identity_digest: str,
) -> tuple[RelocationSourceTrain, ...]:
    manifest_root = root / ".maintenance-state" / "durable-change-trains"
    _real_directory(root / ".maintenance-state", label="maintenance state")
    _real_directory(manifest_root, label="durable change-train state")
    if (manifest_root / ".bootstrap").exists() or (manifest_root / ".bootstrap.pending").exists():
        raise ArchiveRootRelocationError("archive-root relocation does not support fresh-bootstrap train authority")
    manifests = _released_train_manifests_by_target(manifest_root, ArchiveTier.SOURCE)
    if not manifests:
        raise ArchiveRootRelocationError("archive-root relocation requires released source train evidence")
    try:
        _require_released_train_chain(
            ArchiveTier.SOURCE,
            manifests,
            current_version=source_version,
        )
    except DurableChangeTrainError as exc:
        raise ArchiveRootRelocationError("archive-root relocation source train chain is not released") from exc
    expected_targets = set(range(DURABLE_MIGRATION_ADOPTION_FLOORS[ArchiveTier.SOURCE] + 1, source_version + 1))
    if set(manifests) != expected_targets:
        raise ArchiveRootRelocationError("archive-root relocation found an unexpected source train target")
    trains: list[RelocationSourceTrain] = []
    for _target, train in sorted(manifests.items()):
        path = manifest_root / f"source-{train.slot:03d}.json"
        _real_file(path, label="source train manifest")
        if train.state is not DurableChangeTrainState.RELEASED or train.apply_evidence is None:
            raise ArchiveRootRelocationError(f"source train is not released: {path}")
        continuity_refs = tuple(
            ref.removeprefix("proof:source-continuity-refresh:")
            for ref in train.proof_refs
            if ref.startswith("proof:source-continuity-refresh:")
        )
        if (
            train.target_version == source_version
            and train.source_continuity_evidence is None
            and train.apply_evidence.post.content_sha256 != source_content_sha256
        ):
            raise ArchiveRootRelocationError(
                "archive-root relocation requires a typed source-continuity refresh for the live source train; "
                "the released source train still carries stale source content authority"
            )
        if train.source_continuity_evidence is not None:
            try:
                _validate_source_continuity_refresh_receipt(root, train)
            except DurableChangeTrainError as exc:
                raise ArchiveRootRelocationError(
                    "archive-root relocation source continuity authority is invalid"
                ) from exc
        trains.append(
            RelocationSourceTrain(
                path=str(path),
                before_revision=train.revision,
                before_manifest_sha256=_sha256_file(path),
                before_archive_identity_digest=train.apply_evidence.post.archive_identity_digest,
                after_archive_identity_digest=after_identity_digest,
                requires_rebind=train.apply_evidence.post.archive_identity_digest != after_identity_digest,
                source_continuity_receipt_digests=continuity_refs,
            )
        )
        if trains[-1].before_archive_identity_digest == after_identity_digest and not (
            train.target_version == source_version and train.source_continuity_evidence is not None
        ):
            raise ArchiveRootRelocationError(
                f"released source train already carries the current archive identity: {path}"
            )
    return tuple(trains)


def _authenticated_identity(payload: object, *, label: str) -> tuple[int, int]:
    if not isinstance(payload, dict):
        raise ArchiveRootRelocationError(f"backup lacks authenticated {label} identity")
    device = payload.get("device")
    inode = payload.get("inode")
    if type(device) is not int or type(inode) is not int:
        raise ArchiveRootRelocationError(f"backup lacks authenticated {label} device/inode")
    return device, inode


def _authenticated_old_tier_identities(manifest: dict[str, object]) -> dict[str, tuple[int, int]]:
    fingerprints = manifest.get("tier_source_fingerprints")
    if not isinstance(fingerprints, dict):
        raise ArchiveRootRelocationError("backup lacks authenticated tier identity inventory")
    return {
        tier.value: _authenticated_identity(fingerprints.get(f"{tier.value}.db"), label=f"{tier.value} tier")
        for tier in ArchiveTier
    }


def _require_identity_continuity(*, old_device: int, old_inode: int, device: int, inode: int, label: str) -> None:
    if (old_device, old_inode) != (device, inode):
        raise ArchiveRootRelocationError(
            f"archive-root relocation requires {label} device/inode continuity; a copied archive is not accepted"
        )


def _check_backup_against_live(
    root: Path,
    *,
    manifest: dict[str, object],
    receipt: dict[str, object],
    snapshots: tuple[RelocationTierEvidence, ...],
) -> None:
    fingerprints = manifest["tier_source_fingerprints"]
    artifacts = receipt["tier_artifacts"]
    assert isinstance(fingerprints, dict)
    assert isinstance(artifacts, list)
    by_tier = {str(item["tier"]): item for item in artifacts if isinstance(item, dict) and "tier" in item}
    for snapshot in snapshots:
        filename = f"{snapshot.tier}.db"
        fingerprint = fingerprints.get(filename)
        artifact = by_tier.get(snapshot.tier)
        if not isinstance(fingerprint, dict) or not isinstance(artifact, dict):
            raise ArchiveRootRelocationError(f"backup lacks {filename} evidence")
        fields = {
            "device": snapshot.old_device,
            "inode": snapshot.old_inode,
            "size_bytes": snapshot.size_bytes,
            "sha256": snapshot.sha256,
            "user_version": snapshot.user_version,
        }
        if any(fingerprint.get(key) != value for key, value in fields.items()):
            raise ArchiveRootRelocationError(f"backup bytes/version differ from relocated {filename}")
        artifact_fingerprint = artifact.get("source_fingerprint")
        if not isinstance(artifact_fingerprint, dict) or any(
            artifact_fingerprint.get(key) != value for key, value in fields.items()
        ):
            raise ArchiveRootRelocationError(f"backup receipt differs from relocated {filename}")
        _require_identity_continuity(
            old_device=snapshot.old_device,
            old_inode=snapshot.old_inode,
            device=snapshot.device,
            inode=snapshot.inode,
            label=filename,
        )
    _reject_sidecars(root)


def prepare_archive_root_relocation(
    *,
    old_root: Path,
    new_root: Path,
    backup_manifest: Path,
    stopped_daemon_evidence_ref: str,
    single_writer_evidence_ref: str,
) -> ArchiveRootRelocationPlan:
    """Capture immutable, read-only evidence for the one root transition."""
    old_configured = old_root.absolute()
    old_resolved = old_root.resolve(strict=False)
    new_configured = new_root.absolute()
    new_resolved = _real_directory(new_root, label="new archive root")
    if old_resolved == new_resolved:
        raise ArchiveRootRelocationError("archive-root relocation requires distinct old and new roots")
    _reject_sidecars(new_resolved)
    try:
        manifest_path, receipt_path, manifest, receipt = validate_full_evidence_backup_for_archive_root_relocation(
            backup_manifest,
            old_configured_root=old_configured,
            old_archive_root=old_resolved,
        )
    except MigrationError as exc:
        raise ArchiveRootRelocationError(str(exc)) from exc
    old_root_device, old_root_inode = _authenticated_identity(
        manifest.get("archive_root_source_identity"), label="archive root"
    )
    old_tier_identities = _authenticated_old_tier_identities(manifest)
    snapshots = tuple(
        _tier_snapshot(
            new_resolved,
            tier,
            old_device=old_tier_identities[tier.value][0],
            old_inode=old_tier_identities[tier.value][1],
        )
        for tier in ArchiveTier
    )
    _check_backup_against_live(new_resolved, manifest=manifest, receipt=receipt, snapshots=snapshots)
    location_identity = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(new_resolved))
    source_identity_digest = hashlib.sha256(location_identity.tier("source").stable_id.encode()).hexdigest()
    source_version = next(item.user_version for item in snapshots if item.tier == "source")
    source_content_sha256 = next(item.content_sha256 for item in snapshots if item.tier == "source")
    trains = _source_trains(
        new_resolved,
        source_version=source_version,
        source_content_sha256=source_content_sha256,
        after_identity_digest=source_identity_digest,
    )
    root_metadata = new_resolved.stat()
    _require_identity_continuity(
        old_device=old_root_device,
        old_inode=old_root_inode,
        device=root_metadata.st_dev,
        inode=root_metadata.st_ino,
        label="root",
    )
    return _sealed_plan(
        old_configured_root=str(old_configured),
        old_resolved_root=str(old_resolved),
        old_root_device=old_root_device,
        old_root_inode=old_root_inode,
        new_configured_root=str(new_configured),
        new_resolved_root=str(new_resolved),
        new_root_device=root_metadata.st_dev,
        new_root_inode=root_metadata.st_ino,
        backup_manifest_path=str(manifest_path),
        backup_manifest_sha256=_sha256_file(manifest_path),
        backup_receipt_path=str(receipt_path),
        backup_receipt_sha256=_sha256_file(receipt_path),
        backup_profile="full_evidence",
        backup_tier_inventory=tuple(sorted(f"{tier}.db" for tier in _TIER_NAMES)),
        tiers=snapshots,
        source_trains=trains,
        stopped_daemon_evidence_ref=stopped_daemon_evidence_ref,
        single_writer_evidence_ref=single_writer_evidence_ref,
        bound_confirmation="archive-root-relocation",
    )


def write_archive_root_relocation_plan(plan: ArchiveRootRelocationPlan, output: Path) -> None:
    _verify_plan(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def load_archive_root_relocation_plan(path: Path) -> ArchiveRootRelocationPlan:
    try:
        plan = ArchiveRootRelocationPlan.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArchiveRootRelocationError(f"invalid archive-root relocation plan: {path}") from exc
    _verify_plan(plan)
    return plan


def _receipt_path(root: Path, plan: ArchiveRootRelocationPlan) -> Path:
    return root / ".maintenance-state" / "archive-root-relocations" / f"{plan.plan_sha256}.json"


def _receipt_directory_binding(path: Path) -> tuple[Path, str]:
    state_root = path.parent.parent
    if state_root.name != ".maintenance-state" or path.suffix != ".json":
        raise ArchiveRootRelocationError(f"invalid archive-root relocation receipt path: {path}")
    return state_root.parent, path.parent.name


def _decode_receipt(encoded: bytes, *, path: Path) -> ArchiveRootRelocationReceipt:
    try:
        receipt = ArchiveRootRelocationReceipt.model_validate_json(encoded)
    except ValueError as exc:
        raise ArchiveRootRelocationError(f"invalid archive-root relocation receipt: {path}") from exc
    _verify_receipt(receipt)
    return receipt


def _load_receipt_for_update(path: Path) -> ArchiveRootRelocationReceipt | None:
    root, directory_name = _receipt_directory_binding(path)
    try:
        with maintenance_receipt_directory(root, directory_name) as directory_fd:
            encoded = read_optional_receipt(directory_fd, path.name)
    except MaintenanceReceiptPathError as exc:
        raise ArchiveRootRelocationError(f"unsafe archive-root relocation receipt path: {path}") from exc
    return None if encoded is None else _decode_receipt(encoded, path=path)


def _write_receipt(path: Path, receipt: ArchiveRootRelocationReceipt, *, expected: str | None) -> None:
    _verify_receipt(receipt)
    root, directory_name = _receipt_directory_binding(path)
    try:
        with maintenance_receipt_directory(root, directory_name) as directory_fd:
            current_bytes = read_optional_receipt(directory_fd, path.name)
            if current_bytes is not None:
                current = _decode_receipt(current_bytes, path=path)
                if current.receipt_sha256 != expected:
                    raise ArchiveRootRelocationError("archive-root relocation receipt CAS state changed")
            elif expected is not None:
                raise ArchiveRootRelocationError("archive-root relocation receipt disappeared")
            encoded = (json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode()
            atomic_replace_receipt(directory_fd, path.name, encoded)
    except MaintenanceReceiptPathError as exc:
        raise ArchiveRootRelocationError(f"unsafe archive-root relocation receipt path: {path}") from exc


def load_archive_root_relocation_receipt(path: Path) -> ArchiveRootRelocationReceipt:
    try:
        root, directory_name = _receipt_directory_binding(path)
        with existing_maintenance_receipt_directory(root, directory_name) as directory_fd:
            if directory_fd is None:
                raise ArchiveRootRelocationError(f"invalid archive-root relocation receipt: {path}")
            encoded = read_optional_receipt(directory_fd, path.name)
    except MaintenanceReceiptPathError as exc:
        raise ArchiveRootRelocationError(f"unsafe archive-root relocation receipt path: {path}") from exc
    if encoded is None:
        raise ArchiveRootRelocationError(f"invalid archive-root relocation receipt: {path}")
    return _decode_receipt(encoded, path=path)


def assert_no_prepared_archive_root_relocation(root: Path) -> None:
    try:
        with existing_maintenance_receipt_directory(root, "archive-root-relocations") as directory_fd:
            if directory_fd is None:
                return
            receipts = tuple(iter_pinned_receipts(directory_fd))
    except MaintenanceReceiptPathError as exc:
        raise ArchiveRootRelocationError(f"unsafe archive-root relocation receipt directory: {exc}") from exc
    for filename, encoded in receipts:
        receipt = _decode_receipt(encoded, path=root / ".maintenance-state" / "archive-root-relocations" / filename)
        if receipt.state == "prepared":
            raise ArchiveRootRelocationError(
                "archive-root relocation is prepared but incomplete; rerun " + receipt.resume_command
            )


def _revalidate_plan_live_state(
    root: Path,
    plan: ArchiveRootRelocationPlan,
) -> None:
    """Recheck every immutable plan binding while allowing CAS resume states."""
    root_metadata = root.stat()
    if (root_metadata.st_dev, root_metadata.st_ino) != (plan.new_root_device, plan.new_root_inode):
        raise ArchiveRootRelocationError("archive-root relocation configured root identity changed")
    _reject_sidecars(root)
    try:
        manifest_path, receipt_path, manifest, receipt = validate_full_evidence_backup_for_archive_root_relocation(
            Path(plan.backup_manifest_path),
            old_configured_root=Path(plan.old_configured_root),
            old_archive_root=Path(plan.old_resolved_root),
        )
    except MigrationError as exc:
        raise ArchiveRootRelocationError(str(exc)) from exc
    if (
        str(manifest_path) != plan.backup_manifest_path
        or str(receipt_path) != plan.backup_receipt_path
        or _sha256_file(manifest_path) != plan.backup_manifest_sha256
        or _sha256_file(receipt_path) != plan.backup_receipt_sha256
    ):
        raise ArchiveRootRelocationError("archive-root relocation backup authority changed")
    expected_inventory = tuple(sorted(f"{tier}.db" for tier in _TIER_NAMES))
    if plan.backup_tier_inventory != expected_inventory:
        raise ArchiveRootRelocationError("archive-root relocation plan tier inventory changed")
    if _authenticated_identity(manifest.get("archive_root_source_identity"), label="archive root") != (
        plan.old_root_device,
        plan.old_root_inode,
    ):
        raise ArchiveRootRelocationError("archive-root relocation old root identity authority changed")
    old_tiers = {item.tier: (item.old_device, item.old_inode) for item in plan.tiers}
    if len(plan.tiers) != len(ArchiveTier) or set(old_tiers) != {tier.value for tier in ArchiveTier}:
        raise ArchiveRootRelocationError("archive-root relocation plan tier evidence is incomplete")
    snapshots = tuple(
        _tier_snapshot(
            root,
            tier,
            old_device=old_tiers[tier.value][0],
            old_inode=old_tiers[tier.value][1],
        )
        for tier in ArchiveTier
    )
    if snapshots != plan.tiers:
        raise ArchiveRootRelocationError("archive-root relocation tier evidence changed")
    _check_backup_against_live(root, manifest=manifest, receipt=receipt, snapshots=snapshots)
    pending_receipt = _load_receipt_for_update(_receipt_path(root, plan))
    allowed_pending_relocation_receipt_sha256 = (
        pending_receipt.receipt_sha256 if pending_receipt is not None and pending_receipt.state == "prepared" else None
    )
    for item in plan.source_trains:
        path = Path(item.path)
        train = load_durable_change_train_manifest(path)
        continuity_refs = tuple(
            ref.removeprefix("proof:source-continuity-refresh:")
            for ref in train.proof_refs
            if ref.startswith("proof:source-continuity-refresh:")
        )
        before = _sha256_file(path) == item.before_manifest_sha256
        after = (
            train.revision == item.before_revision + (1 if item.requires_rebind else 0)
            and train.apply_evidence is not None
            and train.apply_evidence.post.archive_identity_digest == item.after_archive_identity_digest
            and (
                train.source_continuity_evidence is None
                or train.source_continuity_evidence.archive_identity_digest == item.after_archive_identity_digest
            )
        )
        if not before and not after:
            raise ArchiveRootRelocationError(f"archive-root relocation manifest changed: {path}")
        if before and continuity_refs != item.source_continuity_receipt_digests:
            raise ArchiveRootRelocationError(f"archive-root relocation continuity receipts changed: {path}")
        if train.source_continuity_evidence is not None:
            try:
                _validate_source_continuity_refresh_receipt(
                    root,
                    train,
                    allowed_pending_relocation_receipt_sha256=allowed_pending_relocation_receipt_sha256,
                )
            except DurableChangeTrainError as exc:
                raise ArchiveRootRelocationError(
                    f"archive-root relocation continuity receipt is invalid: {path}"
                ) from exc


def _require_offline_apply_boundary(root: Path) -> None:
    reason = offline_writer_block_reason(Config(archive_root=root, render_root=render_root(), sources=[]))
    if reason is not None:
        raise ArchiveRootRelocationError(f"archive-root relocation requires the daemon to be stopped; {reason}")


def apply_archive_root_relocation(
    *,
    root: Path,
    plan: ArchiveRootRelocationPlan,
    authorization: str,
) -> ArchiveRootRelocationResult:
    """Acquire real offline ownership before any receipt or manifest publication."""
    resolved = _real_directory(root, label="configured archive root")
    try:
        with OwnedArchiveLocation.acquire(
            ArchiveLocation.resolve(resolved),
            owner_id=f"archive-root-relocation:{os.getpid()}",
        ):
            _require_offline_apply_boundary(resolved)
            return _apply_archive_root_relocation_locked(
                root=resolved,
                plan=plan,
                authorization=authorization,
            )
    except ArchiveOwnershipError as exc:
        raise ArchiveRootRelocationError(
            "archive-root relocation could not acquire exclusive archive ownership"
        ) from exc


def _apply_archive_root_relocation_locked(
    *,
    root: Path,
    plan: ArchiveRootRelocationPlan,
    authorization: str,
) -> ArchiveRootRelocationResult:
    """CAS-rewrite only released source manifests under the owned offline boundary."""
    _verify_plan(plan)
    if authorization != plan.plan_sha256 or plan.bound_confirmation != "archive-root-relocation":
        raise ArchiveRootRelocationError("archive-root relocation authorization does not bind this plan")
    if str(root.absolute()) != plan.new_configured_root or str(root) != plan.new_resolved_root:
        raise ArchiveRootRelocationError("archive-root relocation plan is bound to a different configured root")
    _revalidate_plan_live_state(root, plan)
    receipt_path = _receipt_path(root, plan)
    command = (
        f"POLYLOGUE_ARCHIVE_ROOT={plan.new_configured_root} polylogue ops maintenance archive-root-relocation "
        f"apply --plan <plan.json> --authorize {plan.plan_sha256} --output-format json"
    )
    before_hashes = tuple(item.before_manifest_sha256 for item in plan.source_trains)
    receipt = _sealed_receipt(
        state="prepared",
        revision=0,
        plan_sha256=plan.plan_sha256,
        authorization=authorization,
        manifest_before_sha256=before_hashes,
        manifest_after_sha256=(),
        resume_command=command,
    )
    existing_receipt = _load_receipt_for_update(receipt_path)
    if existing_receipt is not None:
        receipt = existing_receipt
        if receipt.plan_sha256 != plan.plan_sha256 or receipt.authorization != authorization:
            raise ArchiveRootRelocationError("archive-root relocation receipt belongs to another plan")
        if receipt.state == "committed":
            if tuple(_sha256_file(Path(item.path)) for item in plan.source_trains) != receipt.manifest_after_sha256:
                raise ArchiveRootRelocationError("archive-root relocation committed receipt does not match manifests")
            return ArchiveRootRelocationResult(
                state="committed",
                plan_sha256=plan.plan_sha256,
                receipt_path=str(receipt_path),
                changed_manifests=tuple(item.path for item in plan.source_trains),
            )
    else:
        _write_receipt(receipt_path, receipt, expected=None)
    after_hashes: list[str] = []
    for item in plan.source_trains:
        path = Path(item.path)
        train = load_durable_change_train_manifest(path)
        actual_hash = _sha256_file(path)
        if actual_hash == item.before_manifest_sha256 and item.requires_rebind:
            continuity_transition_ref = None
            if train.source_continuity_evidence is not None:
                transition_digest = write_source_continuity_relocation_transition(
                    root,
                    train=train,
                    archive_identity_digest=item.after_archive_identity_digest,
                    relocation_plan_sha256=plan.plan_sha256,
                    relocation_receipt_sha256=receipt.receipt_sha256,
                )
                continuity_transition_ref = f"proof:source-continuity-relocation:{transition_digest}"
            updated = rebind_released_source_train_archive_identity(
                train,
                archive_identity_digest=item.after_archive_identity_digest,
                proof_refs=tuple(
                    ref
                    for ref in (
                        f"proof:archive-root-relocation:{receipt.receipt_sha256}",
                        continuity_transition_ref,
                    )
                    if ref is not None
                ),
            )
            write_durable_change_train_manifest(path, updated, expected_revision=item.before_revision)
        elif (
            train.revision != item.before_revision + (1 if item.requires_rebind else 0)
            or train.apply_evidence is None
            or train.apply_evidence.post.archive_identity_digest != item.after_archive_identity_digest
        ):
            raise ArchiveRootRelocationError(
                f"archive-root relocation manifest is neither exact before nor after: {path}"
            )
        after_hashes.append(_sha256_file(path))
    committed = _sealed_receipt(
        state="committed",
        revision=1,
        plan_sha256=plan.plan_sha256,
        authorization=authorization,
        manifest_before_sha256=before_hashes,
        manifest_after_sha256=tuple(after_hashes),
        resume_command=command,
        prepared_receipt_sha256=receipt.receipt_sha256,
    )
    _write_receipt(receipt_path, committed, expected=receipt.receipt_sha256)
    return ArchiveRootRelocationResult(
        state="committed",
        plan_sha256=plan.plan_sha256,
        receipt_path=str(receipt_path),
        changed_manifests=tuple(item.path for item in plan.source_trains),
    )

"""One explicit offline transition for an inode-preserving archive-root move."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import (
    DurableChangeTrainState,
    load_durable_change_train_manifest,
    rebind_released_source_train_archive_identity,
    write_durable_change_train_manifest,
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


class ArchiveRootRelocationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["polylogue.archive-root-relocation-plan.v1"] = PLAN_FORMAT
    old_configured_root: str
    old_resolved_root: str
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
    payload = {"format": PLAN_FORMAT, **values, "plan_sha256": ""}
    payload["plan_sha256"] = _canonical_sha256({key: value for key, value in payload.items() if key != "plan_sha256"})
    return ArchiveRootRelocationPlan.model_validate(payload)


def _sealed_receipt(**values: object) -> ArchiveRootRelocationReceipt:
    payload = {"format": RECEIPT_FORMAT, **values, "receipt_sha256": ""}
    payload["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    return ArchiveRootRelocationReceipt.model_validate(payload)


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


def _tier_snapshot(root: Path, tier: ArchiveTier) -> RelocationTierEvidence:
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
    accepted_before_digests: frozenset[str],
    after_identity_digest: str,
) -> tuple[RelocationSourceTrain, ...]:
    manifest_root = root / ".maintenance-state" / "durable-change-trains"
    _real_directory(root / ".maintenance-state", label="maintenance state")
    _real_directory(manifest_root, label="durable change-train state")
    if (manifest_root / ".bootstrap").exists() or (manifest_root / ".bootstrap.pending").exists():
        raise ArchiveRootRelocationError("archive-root relocation does not support fresh-bootstrap train authority")
    paths = tuple(sorted(manifest_root.glob("source-*.json")))
    if not paths:
        raise ArchiveRootRelocationError("archive-root relocation requires released source train evidence")
    trains: list[RelocationSourceTrain] = []
    for path in paths:
        _real_file(path, label="source train manifest")
        train = load_durable_change_train_manifest(path)
        if train.state is not DurableChangeTrainState.RELEASED or train.apply_evidence is None:
            raise ArchiveRootRelocationError(f"source train is not released: {path}")
        if train.source_continuity_evidence is not None:
            raise ArchiveRootRelocationError(
                "archive-root relocation does not support source-continuity train authority"
            )
        trains.append(
            RelocationSourceTrain(
                path=str(path),
                before_revision=train.revision,
                before_manifest_sha256=_sha256_file(path),
                before_archive_identity_digest=train.apply_evidence.post.archive_identity_digest,
                after_archive_identity_digest=after_identity_digest,
            )
        )
        if trains[-1].before_archive_identity_digest not in accepted_before_digests:
            raise ArchiveRootRelocationError(
                f"released source train does not independently prove the relocated source inode: {path}"
            )
    return tuple(trains)


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
        fields = {"size_bytes": snapshot.size_bytes, "sha256": snapshot.sha256, "user_version": snapshot.user_version}
        if any(fingerprint.get(key) != value for key, value in fields.items()):
            raise ArchiveRootRelocationError(f"backup bytes/version differ from relocated {filename}")
        artifact_fingerprint = artifact.get("source_fingerprint")
        if not isinstance(artifact_fingerprint, dict) or any(
            artifact_fingerprint.get(key) != value for key, value in fields.items()
        ):
            raise ArchiveRootRelocationError(f"backup receipt differs from relocated {filename}")
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
            backup_manifest, old_archive_root=old_resolved
        )
    except MigrationError as exc:
        raise ArchiveRootRelocationError(str(exc)) from exc
    snapshots = tuple(_tier_snapshot(new_resolved, tier) for tier in ArchiveTier)
    _check_backup_against_live(new_resolved, manifest=manifest, receipt=receipt, snapshots=snapshots)
    location_identity = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(new_resolved))
    source_identity_digest = hashlib.sha256(location_identity.tier("source").stable_id.encode()).hexdigest()
    old_location_identity = replace(location_identity, configured_root=old_resolved)
    trains = _source_trains(
        new_resolved,
        accepted_before_digests=frozenset({source_identity_digest, old_location_identity.authority_identity_digest}),
        after_identity_digest=source_identity_digest,
    )
    root_metadata = new_resolved.stat()
    return _sealed_plan(
        old_configured_root=str(old_configured),
        old_resolved_root=str(old_resolved),
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


def _write_receipt(path: Path, receipt: ArchiveRootRelocationReceipt, *, expected: str | None) -> None:
    _verify_receipt(receipt)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _real_directory(path.parent, label="archive-root relocation receipt directory")
    if path.exists():
        current = load_archive_root_relocation_receipt(path)
        if current.receipt_sha256 != expected:
            raise ArchiveRootRelocationError("archive-root relocation receipt CAS state changed")
    elif expected is not None:
        raise ArchiveRootRelocationError("archive-root relocation receipt disappeared")
    encoded = (json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def load_archive_root_relocation_receipt(path: Path) -> ArchiveRootRelocationReceipt:
    _real_file(path, label="archive-root relocation receipt")
    try:
        receipt = ArchiveRootRelocationReceipt.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArchiveRootRelocationError(f"invalid archive-root relocation receipt: {path}") from exc
    _verify_receipt(receipt)
    return receipt


def assert_no_prepared_archive_root_relocation(root: Path) -> None:
    receipt_root = root / ".maintenance-state" / "archive-root-relocations"
    if not receipt_root.exists():
        return
    _real_directory(receipt_root, label="archive-root relocation receipt directory")
    for path in sorted(receipt_root.glob("*.json")):
        receipt = load_archive_root_relocation_receipt(path)
        if receipt.state == "prepared":
            raise ArchiveRootRelocationError(
                "archive-root relocation is prepared but incomplete; rerun " + receipt.resume_command
            )


def _revalidate_plan_live_state(
    root: Path,
    plan: ArchiveRootRelocationPlan,
    *,
    stopped_daemon_evidence_ref: str,
    single_writer_evidence_ref: str,
) -> None:
    """Recheck every immutable plan binding while allowing CAS resume states."""
    if stopped_daemon_evidence_ref != plan.stopped_daemon_evidence_ref:
        raise ArchiveRootRelocationError("archive-root relocation stopped-daemon evidence changed")
    if single_writer_evidence_ref != plan.single_writer_evidence_ref:
        raise ArchiveRootRelocationError("archive-root relocation single-writer evidence changed")
    root_metadata = root.stat()
    if (root_metadata.st_dev, root_metadata.st_ino) != (plan.new_root_device, plan.new_root_inode):
        raise ArchiveRootRelocationError("archive-root relocation configured root identity changed")
    _reject_sidecars(root)
    try:
        manifest_path, receipt_path, manifest, receipt = validate_full_evidence_backup_for_archive_root_relocation(
            Path(plan.backup_manifest_path), old_archive_root=Path(plan.old_resolved_root)
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
    snapshots = tuple(_tier_snapshot(root, tier) for tier in ArchiveTier)
    if snapshots != plan.tiers:
        raise ArchiveRootRelocationError("archive-root relocation tier evidence changed")
    _check_backup_against_live(root, manifest=manifest, receipt=receipt, snapshots=snapshots)
    for item in plan.source_trains:
        path = Path(item.path)
        train = load_durable_change_train_manifest(path)
        before = _sha256_file(path) == item.before_manifest_sha256
        after = (
            train.revision == item.before_revision + 1
            and train.apply_evidence is not None
            and train.apply_evidence.post.archive_identity_digest == item.after_archive_identity_digest
        )
        if not before and not after:
            raise ArchiveRootRelocationError(f"archive-root relocation manifest changed: {path}")


def apply_archive_root_relocation(
    *,
    root: Path,
    plan: ArchiveRootRelocationPlan,
    authorization: str,
    stopped_daemon_evidence_ref: str,
    single_writer_evidence_ref: str,
) -> ArchiveRootRelocationResult:
    """CAS-rewrite only released source manifests, never SQLite/archive bytes."""
    _verify_plan(plan)
    if authorization != plan.plan_sha256 or plan.bound_confirmation != "archive-root-relocation":
        raise ArchiveRootRelocationError("archive-root relocation authorization does not bind this plan")
    resolved = _real_directory(root, label="configured archive root")
    if str(root.absolute()) != plan.new_configured_root or str(resolved) != plan.new_resolved_root:
        raise ArchiveRootRelocationError("archive-root relocation plan is bound to a different configured root")
    _revalidate_plan_live_state(
        resolved,
        plan,
        stopped_daemon_evidence_ref=stopped_daemon_evidence_ref,
        single_writer_evidence_ref=single_writer_evidence_ref,
    )
    receipt_path = _receipt_path(resolved, plan)
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
    if receipt_path.exists():
        receipt = load_archive_root_relocation_receipt(receipt_path)
        if receipt.plan_sha256 != plan.plan_sha256 or receipt.authorization != authorization:
            raise ArchiveRootRelocationError("archive-root relocation receipt belongs to another plan")
        if receipt.state == "committed":
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
        if actual_hash == item.before_manifest_sha256:
            updated = rebind_released_source_train_archive_identity(
                train,
                archive_identity_digest=item.after_archive_identity_digest,
                proof_ref=f"proof:archive-root-relocation:{receipt.receipt_sha256}",
            )
            write_durable_change_train_manifest(path, updated, expected_revision=item.before_revision)
        elif (
            train.revision != item.before_revision + 1
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
    )
    _write_receipt(receipt_path, committed, expected=receipt.receipt_sha256)
    return ArchiveRootRelocationResult(
        state="committed",
        plan_sha256=plan.plan_sha256,
        receipt_path=str(receipt_path),
        changed_manifests=tuple(item.path for item in plan.source_trains),
    )

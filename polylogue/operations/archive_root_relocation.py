"""One explicit offline transition for an inode-preserving archive-root move."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sqlite3
import stat
import tempfile
import uuid
from contextlib import closing, suppress
from pathlib import Path
from typing import Literal, cast

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
    ArchiveTierName,
    OwnedArchiveLocation,
    TierFileIdentity,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import (
    DURABLE_MIGRATION_ADOPTION_FLOORS,
    DurableChangeTrainError,
    DurableChangeTrainState,
    _released_train_manifests_by_target,
    _require_released_train_chain,
    _validate_archive_root_relocation_receipts,
    _validate_source_continuity_refresh_receipt,
    load_durable_change_train_manifest,
    rebind_released_durable_train_archive_identity,
    write_durable_change_train_manifest,
    write_source_continuity_relocation_transition,
)
from polylogue.storage.sqlite.migration_runner import (
    DURABLE_MIGRATION_TIERS,
    DurableChangeTrain,
    MigrationError,
    capture_durable_database_evidence,
    capture_durable_schema_inventory,
    durable_change_train_to_payload,
    validate_full_evidence_backup_for_archive_root_relocation,
)
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

PLAN_FORMAT: Literal["polylogue.archive-root-relocation-plan.v4"] = "polylogue.archive-root-relocation-plan.v4"
_LEGACY_PLAN_FORMAT: Literal["polylogue.archive-root-relocation-plan.v3"] = "polylogue.archive-root-relocation-plan.v3"
RECEIPT_FORMAT: Literal["polylogue.archive-root-relocation-receipt.v2"] = "polylogue.archive-root-relocation-receipt.v2"
_LEGACY_RECEIPT_FORMAT: Literal["polylogue.archive-root-relocation-receipt.v1"] = (
    "polylogue.archive-root-relocation-receipt.v1"
)
_TIER_NAMES = tuple(tier.value for tier in ArchiveTier)
_DURABLE_TIER_NAMES = ("source", "user", "audit")
_SIDECARS = ("-wal", "-shm", "-journal")


class ArchiveRootRelocationError(DurableChangeTrainError):
    """The requested root move has no single safe offline transition."""


class RelocationTierEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: ArchiveTierName
    configured_path: str
    resolved_path: str
    backup_device: int
    backup_inode: int
    device: int
    inode: int
    size_bytes: int
    sha256: str
    user_version: int
    schema_inventory_sha256: str
    content_sha256: str
    quick_check: tuple[str, ...]


class RelocationDurableTrain(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: Literal["source", "user", "audit"]
    train_id: str
    path: str
    before_revision: int
    before_manifest_sha256: str
    before_archive_identity_digest: str
    after_archive_identity_digest: str
    requires_rebind: bool
    continuity_receipt_digests: tuple[str, ...]


class RelocationActiveIndexPointer(BaseModel):
    """The active index pointer's old target and its owned destination mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    old_target: str
    new_target: str
    old_resolved_target: str
    new_resolved_target: str
    conventional_symlink_old_target: str | None = None
    conventional_symlink_new_target: str | None = None
    device: int
    inode: int


class RelocationPostMoveWitness(BaseModel):
    """Evidence from a mover that rewrote paths before Polylogue could plan.

    The normal route observes an old-root-owned pointer directly.  A managed
    storage move may instead rewrite the pointer and remove the old path
    first.  This witness does not waive identity checks: it supplies the
    historical device identity and binds the two configured roots so the
    released train and the current inodes must still agree.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["polylogue.archive-root-relocation-post-move-witness.v1"]
    old_configured_root: str
    old_resolved_root: str
    new_configured_root: str
    new_resolved_root: str
    legacy_device: int
    source_inode: int
    evidence_ref: str


class RelocationIndexGenerationSymlink(BaseModel):
    """One generation-owned tier link whose absolute target moves with the root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    before_device: int | None = None
    before_inode: int | None = None
    old_target: str
    new_target: str


class RelocationIndexGeneration(BaseModel):
    """Exact before/after authority for retained index-generation metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generation_id: str
    metadata_path: str
    directory_device: int
    directory_inode: int
    metadata_before_device: int | None = None
    metadata_before_inode: int | None = None
    before_sha256: str
    after_sha256: str
    before_archive_root: str
    after_archive_root: str
    before_index_path: str
    after_index_path: str
    tier_symlinks: tuple[RelocationIndexGenerationSymlink, ...]


class ArchiveRootRelocationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal[
        "polylogue.archive-root-relocation-plan.v3",
        "polylogue.archive-root-relocation-plan.v4",
    ] = PLAN_FORMAT
    old_configured_root: str
    old_resolved_root: str
    backup_root_device: int
    backup_root_inode: int
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
    active_index_pointer: RelocationActiveIndexPointer | None
    post_move_witness: RelocationPostMoveWitness | None = None
    index_generations: tuple[RelocationIndexGeneration, ...]
    durable_trains: tuple[RelocationDurableTrain, ...]
    stopped_daemon_evidence_ref: str
    single_writer_evidence_ref: str
    bound_confirmation: str
    plan_sha256: str


class ArchiveRootRelocationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal[
        "polylogue.archive-root-relocation-receipt.v1",
        "polylogue.archive-root-relocation-receipt.v2",
    ] = RECEIPT_FORMAT
    state: Literal["prepared", "committed"]
    revision: int
    plan_sha256: str
    authorization: str
    manifest_before_sha256: tuple[str, ...]
    manifest_after_sha256: tuple[str, ...]
    active_index_pointer_old_target: str | None = None
    active_index_pointer_new_target: str | None = None
    active_index_pointer_new_resolved_target: str | None = None
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


def _receipt_checksum_payload(receipt: ArchiveRootRelocationReceipt) -> dict[str, object]:
    """Render the exact field set that the receipt format originally sealed."""
    payload = receipt.model_dump(exclude={"receipt_sha256"}, mode="json")
    if receipt.format == _LEGACY_RECEIPT_FORMAT:
        return {
            field: payload[field]
            for field in (
                "format",
                "state",
                "revision",
                "plan_sha256",
                "authorization",
                "manifest_before_sha256",
                "manifest_after_sha256",
                "resume_command",
            )
        }
    return payload


def _sealed_plan(**values: object) -> ArchiveRootRelocationPlan:
    plan = ArchiveRootRelocationPlan.model_validate({"format": PLAN_FORMAT, **values, "plan_sha256": ""})
    payload = plan.model_dump(
        mode="json",
        exclude={"plan_sha256"},
        exclude_none=plan.format == _LEGACY_PLAN_FORMAT,
    )
    return plan.model_copy(update={"plan_sha256": _canonical_sha256(payload)})


def _sealed_receipt(**values: object) -> ArchiveRootRelocationReceipt:
    receipt = ArchiveRootRelocationReceipt.model_validate({"format": RECEIPT_FORMAT, **values, "receipt_sha256": ""})
    return receipt.model_copy(update={"receipt_sha256": _canonical_sha256(_receipt_checksum_payload(receipt))})


def _verify_plan(plan: ArchiveRootRelocationPlan) -> None:
    expected = _canonical_sha256(
        plan.model_dump(
            exclude={"plan_sha256"},
            mode="json",
            exclude_none=plan.format == _LEGACY_PLAN_FORMAT,
        )
    )
    if plan.plan_sha256 != expected:
        raise ArchiveRootRelocationError("archive-root relocation plan checksum mismatch")
    if plan.format == PLAN_FORMAT:
        for generation in plan.index_generations:
            if generation.metadata_before_device is None or generation.metadata_before_inode is None:
                raise ArchiveRootRelocationError("archive-root relocation v4 plan lacks generation metadata identity")
            if any(link.before_device is None or link.before_inode is None for link in generation.tier_symlinks):
                raise ArchiveRootRelocationError("archive-root relocation v4 plan lacks generation tier-link identity")


def _verify_receipt(receipt: ArchiveRootRelocationReceipt) -> None:
    expected = _canonical_sha256(_receipt_checksum_payload(receipt))
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
    backup_device: int,
    backup_inode: int,
    active_index_pointer: RelocationActiveIndexPointer | None = None,
) -> RelocationTierEvidence:
    if tier is ArchiveTier.INDEX and active_index_pointer is not None:
        path = Path(active_index_pointer.new_target)
        resolved_path = Path(active_index_pointer.new_resolved_target)
    else:
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
        with closing(sqlite3.connect(f"file:{resolved_path}?mode=ro&immutable=1", uri=True)) as connection:
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
        backup_device=backup_device,
        backup_inode=backup_inode,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size_bytes=metadata.st_size,
        sha256=_sha256_file(resolved_path),
        user_version=user_version,
        schema_inventory_sha256=schema.sha256,
        content_sha256=content_sha256,
        quick_check=quick_check,
    )


def _read_active_index_pointer(root: Path) -> tuple[Path, Path] | None:
    """Read one absolute pointer target without resolving a stale old-root path."""
    pointer = root / ".index-active-pointer"
    try:
        metadata = pointer.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ArchiveRootRelocationError(f"cannot inspect active index pointer: {pointer}") from exc
    try:
        if stat.S_ISLNK(metadata.st_mode):
            raw = os.readlink(pointer)
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            raw = pointer.read_text(encoding="utf-8").strip()
        else:
            raise ArchiveRootRelocationError(
                f"active index pointer is not a regular single-linked file or symlink: {pointer}"
            )
    except (OSError, UnicodeDecodeError) as exc:
        raise ArchiveRootRelocationError(f"cannot read active index pointer: {pointer}") from exc
    target = Path(raw)
    if not target.is_absolute() or target.name != "index.db":
        raise ArchiveRootRelocationError(f"invalid active index pointer target: {target}")
    return pointer, Path(os.path.abspath(target))


def _active_index_pointer_evidence(
    *, old_root: Path, new_root: Path, post_move_witness: RelocationPostMoveWitness | None = None
) -> RelocationActiveIndexPointer | None:
    """Map an old-root-owned target before the relocation can publish it anew."""
    pointer = _read_active_index_pointer(new_root)
    if pointer is None:
        return None
    _pointer_path, old_target = pointer
    try:
        relative_target = old_target.relative_to(old_root)
    except ValueError as exc:
        try:
            moved_relative_target = old_target.relative_to(new_root)
        except ValueError:
            moved_relative_target = None
        if post_move_witness is None or moved_relative_target is None:
            raise ArchiveRootRelocationError(
                "archive-root relocation active index pointer target is not owned by the old root"
            ) from exc
        relative_target = moved_relative_target
        old_target = old_root / relative_target
    new_target = new_root / relative_target
    conventional_old_target: str | None = None
    conventional_new_target: str | None = None
    if new_target.is_symlink():
        conventional_old_target = os.readlink(new_target)
        raw_conventional_target = Path(conventional_old_target)
        if raw_conventional_target.is_absolute():
            try:
                conventional_relative = raw_conventional_target.relative_to(old_root)
            except ValueError as exc:
                try:
                    conventional_relative = raw_conventional_target.relative_to(new_root)
                except ValueError:
                    raise ArchiveRootRelocationError(
                        "archive-root relocation conventional index target is not owned by the old root"
                    ) from exc
            conventional_new_target = str(new_root / conventional_relative)
            new_resolved_target = Path(conventional_new_target).resolve(strict=True)
        else:
            conventional_new_target = conventional_old_target
            new_resolved_target = (new_target.parent / raw_conventional_target).resolve(strict=True)
    else:
        try:
            new_resolved_target = new_target.resolve(strict=True)
        except OSError as exc:
            raise ArchiveRootRelocationError(
                f"cannot resolve mapped active index pointer target: {new_target}"
            ) from exc
    if not new_resolved_target.is_relative_to(new_root):
        raise ArchiveRootRelocationError(
            "archive-root relocation mapped active index pointer target escapes the destination root"
        )
    metadata = _real_file(new_resolved_target, label="mapped active index pointer target")
    old_resolved_target = old_root / new_resolved_target.relative_to(new_root)
    return RelocationActiveIndexPointer(
        old_target=str(old_target),
        new_target=str(new_target),
        old_resolved_target=str(old_resolved_target),
        new_resolved_target=str(new_resolved_target),
        conventional_symlink_old_target=conventional_old_target,
        conventional_symlink_new_target=conventional_new_target,
        device=metadata.st_dev,
        inode=metadata.st_ino,
    )


_INDEX_GENERATION_TIER_LINKS = ("source.db", "user.db", "embeddings.db", "ops.db", "blob")


def _index_generation_metadata_bytes(payload: dict[str, object]) -> bytes:
    """Match ``IndexGenerationStore._write``'s stable persisted representation."""
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def _mapped_generation_path(value: object, *, old_root: Path, new_root: Path, label: str) -> tuple[str, str]:
    if not isinstance(value, str):
        raise ArchiveRootRelocationError(f"archive-root relocation index generation has invalid {label}")
    path = Path(value)
    if not path.is_absolute():
        raise ArchiveRootRelocationError(f"archive-root relocation index generation has non-absolute {label}")
    if path.is_relative_to(old_root):
        return value, str(new_root / path.relative_to(old_root))
    if path.is_relative_to(new_root):
        return value, value
    raise ArchiveRootRelocationError(f"archive-root relocation index generation {label} is not root-owned")


def _index_generations_root(
    root: Path,
    active_index_pointer: RelocationActiveIndexPointer | None,
) -> Path:
    """Return the generation store paired with the plan's canonical index path."""
    canonical_index = Path(active_index_pointer.new_target) if active_index_pointer is not None else root / "index.db"
    try:
        canonical_index.relative_to(root)
    except ValueError as exc:
        raise ArchiveRootRelocationError(
            "archive-root relocation canonical index path escapes the destination root"
        ) from exc
    return canonical_index.parent / ".index-generations"


def _index_generation_evidence(
    *,
    old_root: Path,
    new_root: Path,
    active_index_pointer: RelocationActiveIndexPointer | None,
) -> tuple[RelocationIndexGeneration, ...]:
    """Seal every retained generation's absolute metadata and tier links."""
    generations_root = _index_generations_root(new_root, active_index_pointer)
    if not generations_root.exists() and not generations_root.is_symlink():
        return ()
    _real_directory(generations_root, label="index generations root")
    rows: list[RelocationIndexGeneration] = []
    for generation_root in sorted(generations_root.glob("gen-*")):
        _real_directory(generation_root, label="index generation")
        directory_fd = -1
        try:
            directory_fd = os.open(
                generation_root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            generation_metadata = os.fstat(directory_fd)
            encoded, metadata = _read_pinned_generation_metadata(directory_fd)
            try:
                raw = json.loads(encoded)
            except json.JSONDecodeError as exc:
                raise ArchiveRootRelocationError("cannot read index generation metadata") from exc
            if not isinstance(raw, dict):
                raise ArchiveRootRelocationError("index generation metadata is not an object")
            payload = cast(dict[str, object], raw)
            generation_id = payload.get("generation_id")
            if generation_id != generation_root.name:
                raise ArchiveRootRelocationError("index generation metadata does not bind its directory")
            before_archive_root, after_archive_root = _mapped_generation_path(
                payload.get("archive_root"), old_root=old_root, new_root=new_root, label="archive root"
            )
            if Path(after_archive_root) != new_root:
                raise ArchiveRootRelocationError("index generation metadata archive root is not the destination root")
            before_index_path, after_index_path = _mapped_generation_path(
                payload.get("index_path"), old_root=old_root, new_root=new_root, label="index path"
            )
            if Path(after_index_path) != generation_root / "index.db":
                raise ArchiveRootRelocationError("index generation metadata index path does not bind its generation")
            after_payload = {
                **payload,
                "archive_root": after_archive_root,
                "index_path": after_index_path,
            }
            links: list[RelocationIndexGenerationSymlink] = []
            entries = set(os.listdir(directory_fd))
            for filename in _INDEX_GENERATION_TIER_LINKS:
                if filename not in entries:
                    continue
                link = generation_root / filename
                old_target, link_metadata = _read_pinned_generation_symlink(directory_fd, filename)
                raw_target = Path(old_target)
                if not raw_target.is_absolute():
                    raise ArchiveRootRelocationError("index generation tier link target is not absolute")
                _before, new_target = _mapped_generation_path(
                    old_target,
                    old_root=old_root,
                    new_root=new_root,
                    label=f"{filename} link target",
                )
                if Path(new_target) != new_root / filename:
                    raise ArchiveRootRelocationError("index generation tier link does not bind its archive tier")
                links.append(
                    RelocationIndexGenerationSymlink(
                        path=str(link),
                        before_device=link_metadata.st_dev,
                        before_inode=link_metadata.st_ino,
                        old_target=old_target,
                        new_target=new_target,
                    )
                )
            metadata_path = generation_root / "generation.json"
            rows.append(
                RelocationIndexGeneration(
                    generation_id=generation_id,
                    metadata_path=str(metadata_path),
                    directory_device=generation_metadata.st_dev,
                    directory_inode=generation_metadata.st_ino,
                    metadata_before_device=metadata.st_dev,
                    metadata_before_inode=metadata.st_ino,
                    before_sha256=hashlib.sha256(encoded).hexdigest(),
                    after_sha256=hashlib.sha256(_index_generation_metadata_bytes(after_payload)).hexdigest(),
                    before_archive_root=before_archive_root,
                    after_archive_root=after_archive_root,
                    before_index_path=before_index_path,
                    after_index_path=after_index_path,
                    tier_symlinks=tuple(links),
                )
            )
        except ArchiveRootRelocationError:
            raise
        except OSError as exc:
            raise ArchiveRootRelocationError("cannot pin index generation evidence") from exc
        finally:
            if directory_fd >= 0:
                os.close(directory_fd)
    return tuple(rows)


def _index_generation_payload_for_state(
    item: RelocationIndexGeneration, *, after: bool, encoded: bytes
) -> dict[str, object]:
    try:
        raw = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ArchiveRootRelocationError("archive-root relocation index generation metadata is unreadable") from exc
    if not isinstance(raw, dict):
        raise ArchiveRootRelocationError("archive-root relocation index generation metadata is not an object")
    payload = cast(dict[str, object], raw)
    expected_root = item.after_archive_root if after else item.before_archive_root
    expected_index = item.after_index_path if after else item.before_index_path
    if (
        payload.get("generation_id") != item.generation_id
        or payload.get("archive_root") != expected_root
        or payload.get("index_path") != expected_index
    ):
        raise ArchiveRootRelocationError("archive-root relocation index generation metadata binding changed")
    return payload


def _validate_index_generation_state(
    root: Path,
    items: tuple[RelocationIndexGeneration, ...],
    active_index_pointer: RelocationActiveIndexPointer | None,
    *,
    allow_post_publication: bool,
) -> None:
    generations_root = _index_generations_root(root, active_index_pointer)
    if generations_root.exists() or generations_root.is_symlink():
        _real_directory(generations_root, label="index generations root")
    current_paths = (
        {str(path / "generation.json") for path in generations_root.glob("gen-*")}
        if generations_root.is_dir() and not generations_root.is_symlink()
        else set()
    )
    expected_paths = {item.metadata_path for item in items}
    if current_paths != expected_paths:
        raise ArchiveRootRelocationError("archive-root relocation index generation inventory changed")
    for item in items:
        metadata_path = Path(item.metadata_path)
        directory_fd = _open_pinned_generation_directory(root, item)
        try:
            _pinned_generation_metadata_state(
                directory_fd,
                item,
                allow_post_publication=allow_post_publication,
            )
            generation_root = metadata_path.parent
            for link in item.tier_symlinks:
                path = Path(link.path)
                if path.parent != generation_root or path.name not in _INDEX_GENERATION_TIER_LINKS:
                    raise ArchiveRootRelocationError(
                        "archive-root relocation index generation tier link path binding changed"
                    )
            expected_link_names = {Path(link.path).name for link in item.tier_symlinks}
            current_link_names = set(_INDEX_GENERATION_TIER_LINKS).intersection(os.listdir(directory_fd))
            if current_link_names != expected_link_names:
                raise ArchiveRootRelocationError("archive-root relocation index generation tier inventory changed")
            for link in item.tier_symlinks:
                _pinned_generation_symlink_state(
                    directory_fd,
                    link,
                    allow_post_publication=allow_post_publication,
                )
        finally:
            os.close(directory_fd)


def _open_pinned_generation_directory(root: Path, item: RelocationIndexGeneration) -> int:
    """Open the plan-owned generation directory without following any link."""
    generation_root = Path(item.metadata_path).parent
    try:
        relative = generation_root.relative_to(root)
    except ValueError as exc:
        raise ArchiveRootRelocationError(
            "archive-root relocation index generation directory escapes the destination root"
        ) from exc
    if (
        not relative.parts
        or relative.parts[-1] != item.generation_id
        or Path(item.metadata_path).name != "generation.json"
    ):
        raise ArchiveRootRelocationError("archive-root relocation index generation path binding changed")
    directory_fd = -1
    try:
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        for component in relative.parts:
            if component in {"", ".", ".."}:
                raise ArchiveRootRelocationError("archive-root relocation index generation path is unsafe")
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        metadata = os.fstat(directory_fd)
        if (metadata.st_dev, metadata.st_ino) != (item.directory_device, item.directory_inode):
            raise ArchiveRootRelocationError("archive-root relocation index generation directory identity changed")
        return directory_fd
    except ArchiveRootRelocationError:
        if directory_fd >= 0:
            os.close(directory_fd)
        raise
    except OSError as exc:
        if directory_fd >= 0:
            os.close(directory_fd)
        raise ArchiveRootRelocationError("cannot pin archive-root relocation index generation directory") from exc


def _read_pinned_generation_metadata(directory_fd: int) -> tuple[bytes, os.stat_result]:
    descriptor = -1
    try:
        descriptor = os.open(
            "generation.json",
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_fd,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ArchiveRootRelocationError(
                "archive-root relocation index generation metadata is not a real single-linked file"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read(), metadata
    except ArchiveRootRelocationError:
        raise
    except OSError as exc:
        raise ArchiveRootRelocationError(
            "cannot read pinned archive-root relocation index generation metadata"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_pinned_generation_symlink(directory_fd: int, filename: str) -> tuple[str, os.stat_result]:
    """Read one generation link relative to its pinned directory without following it."""
    try:
        before = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISLNK(before.st_mode):
            raise ArchiveRootRelocationError("archive-root relocation index generation tier link changed")
        target = os.readlink(filename, dir_fd=directory_fd)
        after = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except ArchiveRootRelocationError:
        raise
    except OSError as exc:
        raise ArchiveRootRelocationError("archive-root relocation index generation tier link is unreadable") from exc
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise ArchiveRootRelocationError("archive-root relocation index generation tier link changed while reading")
    return target, after


def _pinned_generation_metadata_state(
    directory_fd: int,
    item: RelocationIndexGeneration,
    *,
    allow_post_publication: bool,
) -> tuple[bytes, bool]:
    """Return metadata bytes and whether they are the exact post-publication state."""
    encoded, metadata = _read_pinned_generation_metadata(directory_fd)
    digest = hashlib.sha256(encoded).hexdigest()
    if digest == item.before_sha256:
        if (
            item.metadata_before_device is not None
            and item.metadata_before_inode is not None
            and (metadata.st_dev, metadata.st_ino) != (item.metadata_before_device, item.metadata_before_inode)
        ):
            raise ArchiveRootRelocationError("archive-root relocation index generation metadata identity changed")
        _index_generation_payload_for_state(item, after=False, encoded=encoded)
        return encoded, False
    if digest == item.after_sha256:
        if not allow_post_publication:
            raise ArchiveRootRelocationError(
                "archive-root relocation generation metadata reached its post-publication state without a prepared receipt"
            )
        _index_generation_payload_for_state(item, after=True, encoded=encoded)
        return encoded, True
    raise ArchiveRootRelocationError("archive-root relocation index generation metadata changed")


def _pinned_generation_symlink_state(
    directory_fd: int,
    link: RelocationIndexGenerationSymlink,
    *,
    allow_post_publication: bool,
) -> bool:
    """Return whether a pinned tier link is in its exact post-publication state."""
    filename = Path(link.path).name
    target, metadata = _read_pinned_generation_symlink(directory_fd, filename)
    if target == link.old_target:
        if (
            link.before_device is not None
            and link.before_inode is not None
            and (metadata.st_dev, metadata.st_ino) != (link.before_device, link.before_inode)
        ):
            raise ArchiveRootRelocationError("archive-root relocation index generation tier link identity changed")
        return False
    if target == link.new_target:
        if not allow_post_publication:
            raise ArchiveRootRelocationError(
                "archive-root relocation generation tier link reached its post-publication state without a prepared receipt"
            )
        return True
    raise ArchiveRootRelocationError("archive-root relocation index generation tier link changed")


def _replace_pinned_generation_metadata(directory_fd: int, payload: bytes) -> None:
    temporary = f".generation.json.relocation-{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ArchiveRootRelocationError(
                    "archive-root relocation index generation metadata write made no progress"
                )
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, "generation.json", src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except ArchiveRootRelocationError:
        raise
    except OSError as exc:
        raise ArchiveRootRelocationError(
            "cannot atomically publish pinned archive-root relocation index generation metadata"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=directory_fd)


def _publish_index_generation_state(
    root: Path,
    items: tuple[RelocationIndexGeneration, ...],
    active_index_pointer: RelocationActiveIndexPointer | None,
) -> None:
    """CAS-publish mapped metadata and links; exact after states are idempotent."""
    _validate_index_generation_state(root, items, active_index_pointer, allow_post_publication=True)
    for item in items:
        directory_fd = _open_pinned_generation_directory(root, item)
        try:
            encoded, metadata_is_after = _pinned_generation_metadata_state(
                directory_fd,
                item,
                allow_post_publication=True,
            )
            if not metadata_is_after and item.before_sha256 != item.after_sha256:
                payload = _index_generation_payload_for_state(item, after=False, encoded=encoded)
                after_payload = {
                    **payload,
                    "archive_root": item.after_archive_root,
                    "index_path": item.after_index_path,
                }
                after_encoded = _index_generation_metadata_bytes(after_payload)
                if hashlib.sha256(after_encoded).hexdigest() != item.after_sha256:
                    raise ArchiveRootRelocationError("archive-root relocation index generation after binding changed")
                _replace_pinned_generation_metadata(directory_fd, after_encoded)
            generation_root = Path(item.metadata_path).parent
            for link in item.tier_symlinks:
                path = Path(link.path)
                if path.parent != generation_root or path.name not in _INDEX_GENERATION_TIER_LINKS:
                    raise ArchiveRootRelocationError(
                        "archive-root relocation index generation tier link path binding changed"
                    )
                link_is_after = _pinned_generation_symlink_state(
                    directory_fd,
                    link,
                    allow_post_publication=True,
                )
                if link_is_after:
                    continue
                if link.old_target == link.new_target:
                    continue
                temporary = f".{path.name}.relocation-{uuid.uuid4().hex}.tmp"
                try:
                    os.symlink(link.new_target, temporary, dir_fd=directory_fd)
                    os.replace(temporary, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
                    os.fsync(directory_fd)
                except ArchiveRootRelocationError:
                    raise
                except OSError as exc:
                    raise ArchiveRootRelocationError(
                        "cannot atomically publish pinned archive-root relocation index generation tier link"
                    ) from exc
                finally:
                    with suppress(FileNotFoundError):
                        os.unlink(temporary, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
    _validate_index_generation_state(root, items, active_index_pointer, allow_post_publication=True)


def _validate_active_index_pointer(
    root: Path,
    pointer: RelocationActiveIndexPointer | None,
) -> None:
    """Accept only the sealed pre-publication or post-publication pointer state."""
    if pointer is None:
        if _read_active_index_pointer(root) is not None:
            raise ArchiveRootRelocationError("archive-root relocation active index pointer appeared after planning")
        return
    current = _read_active_index_pointer(root)
    if current is None:
        raise ArchiveRootRelocationError("archive-root relocation active index pointer disappeared")
    _pointer_path, target = current
    if str(target) not in {pointer.old_target, pointer.new_target}:
        raise ArchiveRootRelocationError("archive-root relocation active index pointer target changed")
    conventional = Path(pointer.new_target)
    if pointer.conventional_symlink_old_target is not None:
        if not conventional.is_symlink():
            raise ArchiveRootRelocationError("archive-root relocation conventional index symlink disappeared")
        conventional_target = os.readlink(conventional)
        if conventional_target not in {
            pointer.conventional_symlink_old_target,
            pointer.conventional_symlink_new_target,
        }:
            raise ArchiveRootRelocationError("archive-root relocation conventional index symlink changed")
    elif conventional.is_symlink():
        raise ArchiveRootRelocationError("archive-root relocation conventional index unexpectedly became a symlink")
    if str(target) == pointer.new_target:
        try:
            resolved = Path(pointer.new_target).resolve(strict=True)
        except OSError as exc:
            raise ArchiveRootRelocationError("archive-root relocation mapped active index pointer disappeared") from exc
        metadata = _real_file(resolved, label="mapped active index pointer target")
        if str(resolved) != pointer.new_resolved_target or (metadata.st_dev, metadata.st_ino) != (
            pointer.device,
            pointer.inode,
        ):
            raise ArchiveRootRelocationError("archive-root relocation mapped active index pointer changed")


def _publish_conventional_index_symlink(root: Path, pointer: RelocationActiveIndexPointer) -> None:
    """Publish the mapped production ``index.db`` symlink before its pointer."""
    old_target = pointer.conventional_symlink_old_target
    new_target = pointer.conventional_symlink_new_target
    if old_target is None or new_target is None or old_target == new_target:
        return
    conventional = Path(pointer.new_target)
    try:
        relative_parent = conventional.parent.relative_to(root)
    except ValueError as exc:
        raise ArchiveRootRelocationError(
            "archive-root relocation conventional index symlink escapes the destination root"
        ) from exc
    directory_fd = -1
    temporary = f".{conventional.name}.relocation-{uuid.uuid4().hex}.tmp"
    try:
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        for component in relative_parent.parts:
            if component in {"", ".", ".."}:
                raise ArchiveRootRelocationError(
                    "archive-root relocation conventional index symlink has an unsafe parent"
                )
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        metadata = os.stat(conventional.name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISLNK(metadata.st_mode):
            raise ArchiveRootRelocationError("archive-root relocation conventional index symlink disappeared")
        current = os.readlink(conventional.name, dir_fd=directory_fd)
        if current == new_target:
            return
        if current != old_target:
            raise ArchiveRootRelocationError("archive-root relocation conventional index symlink changed")
        os.symlink(new_target, temporary, dir_fd=directory_fd)
        os.replace(temporary, conventional.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except ArchiveRootRelocationError:
        raise
    except OSError as exc:
        raise ArchiveRootRelocationError("cannot atomically publish mapped conventional index symlink") from exc
    finally:
        if directory_fd >= 0:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            finally:
                os.close(directory_fd)


def _publish_active_index_pointer(root: Path, pointer: RelocationActiveIndexPointer | None) -> None:
    """Atomically publish the sealed mapped target beneath the owned destination root."""
    if pointer is None:
        return
    _validate_active_index_pointer(root, pointer)
    _publish_conventional_index_symlink(root, pointer)
    current = _read_active_index_pointer(root)
    assert current is not None
    _path, target = current
    if str(target) == pointer.new_target:
        return
    if str(target) != pointer.old_target:
        raise ArchiveRootRelocationError("archive-root relocation active index pointer target changed")
    directory_fd = -1
    temporary = f".index-active-pointer.relocation-{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        root_metadata = root.stat()
        pinned_metadata = os.fstat(directory_fd)
        if (root_metadata.st_dev, root_metadata.st_ino) != (pinned_metadata.st_dev, pinned_metadata.st_ino):
            raise ArchiveRootRelocationError(
                "archive-root relocation destination root changed during pointer publication"
            )
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
        payload = (pointer.new_target + "\n").encode("utf-8")
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("active index pointer write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, ".index-active-pointer", src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError as exc:
        raise ArchiveRootRelocationError("cannot atomically publish mapped active index pointer") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_fd >= 0:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            finally:
                os.close(directory_fd)
    _validate_active_index_pointer(root, pointer)


def _durable_trains(
    root: Path,
    *,
    old_root: Path,
    snapshots: tuple[RelocationTierEvidence, ...],
    post_move_witness: RelocationPostMoveWitness | None = None,
) -> tuple[RelocationDurableTrain, ...]:
    manifest_root = root / ".maintenance-state" / "durable-change-trains"
    _real_directory(root / ".maintenance-state", label="maintenance state")
    _real_directory(manifest_root, label="durable change-train state")
    if (manifest_root / ".bootstrap").exists() or (manifest_root / ".bootstrap.pending").exists():
        raise ArchiveRootRelocationError("archive-root relocation does not support fresh-bootstrap train authority")
    tier_identities = tuple(
        TierFileIdentity(
            item.tier,
            Path(item.configured_path),
            Path(item.resolved_path),
            item.device,
            item.inode,
        )
        for item in snapshots
    )
    index_identity = next(item for item in tier_identities if item.name == "index")
    legacy_active_identity = ArchiveIdentity(
        configured_root=old_root,
        tiers=tier_identities,
        active_generation=index_identity.stable_id,
    ).authority_identity_digest
    configured_location = ArchiveLocation.resolve(root)
    configured_index_identity = configured_location.configured_tier("index")
    legacy_configured_identity = ArchiveIdentity(
        configured_root=old_root,
        tiers=configured_location.configured_tiers,
        active_generation=configured_index_identity.stable_id,
    ).authority_identity_digest
    accepted_legacy_identities = {legacy_active_identity, legacy_configured_identity}
    if post_move_witness is not None:
        witness_tier_identities = tuple(
            TierFileIdentity(
                item.tier,
                old_root / Path(item.configured_path).relative_to(root),
                old_root / Path(item.resolved_path).relative_to(root)
                if Path(item.resolved_path).is_relative_to(root)
                else Path(item.resolved_path),
                post_move_witness.legacy_device,
                item.inode,
            )
            for item in snapshots
        )
        witness_index = next(item for item in witness_tier_identities if item.name == "index")
        accepted_legacy_identities.add(
            ArchiveIdentity(
                configured_root=old_root,
                tiers=witness_tier_identities,
                active_generation=witness_index.stable_id,
            ).authority_identity_digest
        )
    snapshots_by_tier = {item.tier: item for item in snapshots}
    trains: list[RelocationDurableTrain] = []
    for tier in sorted(DURABLE_MIGRATION_TIERS, key=lambda item: item.value):
        snapshot = snapshots_by_tier[tier.value]
        manifests = _released_train_manifests_by_target(manifest_root, tier)
        expected_targets = set(range(DURABLE_MIGRATION_ADOPTION_FLOORS[tier] + 1, snapshot.user_version + 1))
        if tier is ArchiveTier.AUDIT and not manifests:
            # Established archives can carry a verified adopted audit image
            # before source v32 publishes the source-backed continuity head.
            # It has no train manifest to rebind yet; the adoption receipt and
            # full-evidence tier check remain the authority for this narrow
            # transitional state.
            from polylogue.operations.durable_change_train import validate_audit_adoption_receipt

            if validate_audit_adoption_receipt(root) is not None:
                continue
        if set(manifests) != expected_targets:
            raise ArchiveRootRelocationError(f"archive-root relocation found an unexpected {tier.value} train target")
        if manifests:
            try:
                _require_released_train_chain(tier, manifests, current_version=snapshot.user_version)
            except DurableChangeTrainError as exc:
                raise ArchiveRootRelocationError(
                    f"archive-root relocation {tier.value} train chain is not released"
                ) from exc
        tier_identity = next(item for item in tier_identities if item.name == tier.value)
        after_identity_digest = hashlib.sha256(tier_identity.stable_id.encode()).hexdigest()
        for _target, train in sorted(manifests.items()):
            path = manifest_root / f"{tier.value}-{train.slot:03d}.json"
            _real_file(path, label=f"{tier.value} train manifest")
            if train.state is not DurableChangeTrainState.RELEASED or train.apply_evidence is None:
                raise ArchiveRootRelocationError(f"durable train is not released: {path}")
            try:
                _validate_archive_root_relocation_receipts(root, train)
            except DurableChangeTrainError as exc:
                raise ArchiveRootRelocationError(
                    f"archive-root relocation {tier.value} train relocation authority is invalid"
                ) from exc
            continuity_refs = tuple(
                ref.removeprefix("proof:source-continuity-refresh:")
                for ref in train.proof_refs
                if ref.startswith("proof:source-continuity-refresh:")
            )
            if (
                tier is ArchiveTier.SOURCE
                and train.target_version == snapshot.user_version
                and train.source_continuity_evidence is None
                and train.apply_evidence.post.content_sha256 != snapshot.content_sha256
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
            before_identity = train.apply_evidence.post.archive_identity_digest
            if before_identity != after_identity_digest and before_identity not in accepted_legacy_identities:
                raise ArchiveRootRelocationError(
                    f"archive-root relocation {tier.value} train does not authenticate the moved tier identity"
                )
            trains.append(
                RelocationDurableTrain(
                    tier=cast(Literal["source", "user", "audit"], tier.value),
                    train_id=train.train_id,
                    path=str(path),
                    before_revision=train.revision,
                    before_manifest_sha256=_sha256_file(path),
                    before_archive_identity_digest=before_identity,
                    after_archive_identity_digest=after_identity_digest,
                    requires_rebind=before_identity != after_identity_digest,
                    continuity_receipt_digests=continuity_refs,
                )
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


def _authenticated_backup_tier_identities(manifest: dict[str, object]) -> dict[str, tuple[int, int]]:
    fingerprints = manifest.get("tier_source_fingerprints")
    if not isinstance(fingerprints, dict):
        raise ArchiveRootRelocationError("backup lacks authenticated tier identity inventory")
    return {
        tier.value: _authenticated_identity(fingerprints.get(f"{tier.value}.db"), label=f"{tier.value} tier")
        for tier in ArchiveTier
    }


def _require_identity_continuity(*, backup_device: int, backup_inode: int, device: int, inode: int, label: str) -> None:
    if (backup_device, backup_inode) != (device, inode):
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
            "device": snapshot.backup_device,
            "inode": snapshot.backup_inode,
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
            backup_device=snapshot.backup_device,
            backup_inode=snapshot.backup_inode,
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
    post_move_witness: RelocationPostMoveWitness | None = None,
) -> ArchiveRootRelocationPlan:
    """Capture immutable, read-only evidence for the one root transition."""
    old_configured = old_root.absolute()
    old_resolved = old_root.resolve(strict=False)
    new_configured = new_root.absolute()
    new_resolved = _real_directory(new_root, label="new archive root")
    if old_resolved == new_resolved:
        raise ArchiveRootRelocationError("archive-root relocation requires distinct old and new roots")
    if post_move_witness is not None:
        if (
            post_move_witness.old_configured_root != str(old_configured)
            or post_move_witness.old_resolved_root != str(old_resolved)
            or post_move_witness.new_configured_root != str(new_configured)
            or post_move_witness.new_resolved_root != str(new_resolved)
        ):
            raise ArchiveRootRelocationError("archive-root relocation post-move witness root binding changed")
        source_identity = TierFileIdentity.resolve("source", new_resolved / "source.db")
        if source_identity.inode != post_move_witness.source_inode:
            raise ArchiveRootRelocationError("archive-root relocation post-move witness source inode changed")
    _reject_sidecars(new_resolved)
    try:
        manifest_path, receipt_path, manifest, receipt = validate_full_evidence_backup_for_archive_root_relocation(
            backup_manifest,
            backup_configured_root=new_configured,
            backup_archive_root=new_resolved,
        )
    except MigrationError as exc:
        raise ArchiveRootRelocationError(str(exc)) from exc
    backup_root_device, backup_root_inode = _authenticated_identity(
        manifest.get("archive_root_source_identity"), label="archive root"
    )
    backup_tier_identities = _authenticated_backup_tier_identities(manifest)
    active_index_pointer = _active_index_pointer_evidence(
        old_root=old_resolved, new_root=new_resolved, post_move_witness=post_move_witness
    )
    index_generations = _index_generation_evidence(
        old_root=old_resolved,
        new_root=new_resolved,
        active_index_pointer=active_index_pointer,
    )
    snapshots = tuple(
        _tier_snapshot(
            new_resolved,
            tier,
            backup_device=backup_tier_identities[tier.value][0],
            backup_inode=backup_tier_identities[tier.value][1],
            active_index_pointer=active_index_pointer,
        )
        for tier in ArchiveTier
    )
    _check_backup_against_live(new_resolved, manifest=manifest, receipt=receipt, snapshots=snapshots)
    trains = _durable_trains(
        new_resolved,
        old_root=old_resolved,
        snapshots=snapshots,
        post_move_witness=post_move_witness,
    )
    root_metadata = new_resolved.stat()
    _require_identity_continuity(
        backup_device=backup_root_device,
        backup_inode=backup_root_inode,
        device=root_metadata.st_dev,
        inode=root_metadata.st_ino,
        label="root",
    )
    return _sealed_plan(
        old_configured_root=str(old_configured),
        old_resolved_root=str(old_resolved),
        backup_root_device=backup_root_device,
        backup_root_inode=backup_root_inode,
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
        active_index_pointer=active_index_pointer,
        post_move_witness=post_move_witness,
        index_generations=index_generations,
        durable_trains=trains,
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


def _retained_plan_path(root: Path, plan: ArchiveRootRelocationPlan) -> Path:
    return root / ".maintenance-state" / "archive-root-relocation-plans" / f"{plan.plan_sha256}.json"


def _retain_plan(root: Path, plan: ArchiveRootRelocationPlan) -> Path:
    path = _retained_plan_path(root, plan)
    encoded = (json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode()
    try:
        with maintenance_receipt_directory(root, "archive-root-relocation-plans") as directory_fd:
            current = read_optional_receipt(directory_fd, path.name)
            if current is not None and current != encoded:
                raise ArchiveRootRelocationError("archive-root relocation retained plan collision")
            if current is None:
                atomic_replace_receipt(directory_fd, path.name, encoded)
    except MaintenanceReceiptPathError as exc:
        raise ArchiveRootRelocationError("cannot retain archive-root relocation plan") from exc
    return path


def _train_manifest_sha256(train: DurableChangeTrain) -> str:
    payload = durable_change_train_to_payload(train)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


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
        receipt_root = root.resolve(strict=True)
    except OSError as exc:
        raise ArchiveRootRelocationError(f"cannot resolve archive-root relocation archive root: {root}") from exc
    try:
        with existing_maintenance_receipt_directory(receipt_root, "archive-root-relocations") as directory_fd:
            if directory_fd is None:
                return
            receipts = tuple(iter_pinned_receipts(directory_fd))
    except MaintenanceReceiptPathError as exc:
        raise ArchiveRootRelocationError(f"unsafe archive-root relocation receipt directory: {exc}") from exc
    for filename, encoded in receipts:
        receipt = _decode_receipt(
            encoded,
            path=receipt_root / ".maintenance-state" / "archive-root-relocations" / filename,
        )
        if receipt.state == "prepared":
            raise ArchiveRootRelocationError(
                "archive-root relocation is prepared but incomplete; rerun " + receipt.resume_command
            )


def _requires_train_update(item: RelocationDurableTrain) -> bool:
    """Return whether relocation must CAS-revise this released train."""
    return item.requires_rebind or bool(item.continuity_receipt_digests)


def _pointer_receipt_fields(
    pointer: RelocationActiveIndexPointer | None,
) -> tuple[str | None, str | None, str | None]:
    if pointer is None:
        return (None, None, None)
    return (pointer.old_target, pointer.new_target, pointer.new_resolved_target)


def _receipt_pointer_fields_match(
    receipt: ArchiveRootRelocationReceipt, pointer_fields: tuple[str | None, str | None, str | None]
) -> bool:
    """Accept V1's original field set while binding V2 to active-index authority."""
    receipt_fields = (
        receipt.active_index_pointer_old_target,
        receipt.active_index_pointer_new_target,
        receipt.active_index_pointer_new_resolved_target,
    )
    return (
        receipt_fields == (None, None, None)
        if receipt.format == _LEGACY_RECEIPT_FORMAT
        else receipt_fields == pointer_fields
    )


def _has_matching_prepared_publication_receipt(
    plan: ArchiveRootRelocationPlan, receipt: ArchiveRootRelocationReceipt | None
) -> bool:
    """Return whether ``receipt`` proves this plan reached publication.

    A revision-1 receipt replaces the revision-0 preparation receipt in place.
    Its retained digest must therefore recreate that exact preparation receipt,
    rather than merely asserting that a post-publication manifest tuple exists.
    """
    if receipt is None or receipt.state not in {"prepared", "committed"} or receipt.revision < 1:
        return False
    before_hashes = tuple(item.before_manifest_sha256 for item in plan.durable_trains)
    pointer_fields = _pointer_receipt_fields(plan.active_index_pointer)
    if (
        receipt.plan_sha256 != plan.plan_sha256
        or receipt.authorization != plan.plan_sha256
        or receipt.manifest_before_sha256 != before_hashes
        or len(receipt.manifest_after_sha256) != len(plan.durable_trains)
        or (
            receipt.active_index_pointer_old_target,
            receipt.active_index_pointer_new_target,
            receipt.active_index_pointer_new_resolved_target,
        )
        != pointer_fields
        or receipt.prepared_receipt_sha256 is None
    ):
        return False
    preparation = _sealed_receipt(
        format=receipt.format,
        state="prepared",
        revision=0,
        plan_sha256=plan.plan_sha256,
        authorization=plan.plan_sha256,
        manifest_before_sha256=before_hashes,
        manifest_after_sha256=(),
        active_index_pointer_old_target=pointer_fields[0],
        active_index_pointer_new_target=pointer_fields[1],
        active_index_pointer_new_resolved_target=pointer_fields[2],
        resume_command=receipt.resume_command,
    )
    return receipt.prepared_receipt_sha256 == preparation.receipt_sha256


def _has_matching_initial_preparation_receipt(
    plan: ArchiveRootRelocationPlan, receipt: ArchiveRootRelocationReceipt | None
) -> bool:
    """Return whether a retained V3 preparation can safely resume pre-publication.

    V3 has no sealed generation-leaf identities. Its initial receipt proves an
    interrupted apply only while every leaf is still checked as a before-state;
    it never authorizes a post-state leaf or a fresh V3 publication.
    """
    if receipt is None or receipt.state != "prepared" or receipt.revision != 0:
        return False
    before_hashes = tuple(item.before_manifest_sha256 for item in plan.durable_trains)
    pointer_fields = _pointer_receipt_fields(plan.active_index_pointer)
    if (
        receipt.plan_sha256 != plan.plan_sha256
        or receipt.authorization != plan.plan_sha256
        or receipt.manifest_before_sha256 != before_hashes
        or receipt.manifest_after_sha256
        or not _receipt_pointer_fields_match(receipt, pointer_fields)
        or receipt.prepared_receipt_sha256 is not None
    ):
        return False
    preparation = _sealed_receipt(
        format=receipt.format,
        state="prepared",
        revision=0,
        plan_sha256=plan.plan_sha256,
        authorization=plan.plan_sha256,
        manifest_before_sha256=before_hashes,
        manifest_after_sha256=(),
        active_index_pointer_old_target=pointer_fields[0],
        active_index_pointer_new_target=pointer_fields[1],
        active_index_pointer_new_resolved_target=pointer_fields[2],
        resume_command=receipt.resume_command,
    )
    return receipt.receipt_sha256 == preparation.receipt_sha256


def _relocated_train(
    root: Path,
    *,
    plan: ArchiveRootRelocationPlan,
    item: RelocationDurableTrain,
    train: DurableChangeTrain,
    relocation_receipt_sha256: str,
) -> DurableChangeTrain:
    continuity_transition_ref = None
    if train.source_continuity_evidence is not None:
        transition_digest = write_source_continuity_relocation_transition(
            root,
            train=train,
            archive_identity_digest=item.after_archive_identity_digest,
            relocation_plan_sha256=plan.plan_sha256,
            relocation_receipt_sha256=relocation_receipt_sha256,
        )
        continuity_transition_ref = f"proof:source-continuity-relocation:{transition_digest}"
    return rebind_released_durable_train_archive_identity(
        train,
        archive_identity_digest=item.after_archive_identity_digest,
        proof_refs=tuple(
            ref
            for ref in (
                f"proof:archive-root-relocation:{relocation_receipt_sha256}",
                continuity_transition_ref,
            )
            if ref is not None
        ),
    )


def _validate_plan_continuity_binding(
    root: Path,
    *,
    plan: ArchiveRootRelocationPlan,
    item: RelocationDurableTrain,
    train: object,
    before: bool,
    relocation_receipt: ArchiveRootRelocationReceipt | None,
) -> None:
    """Bind resumed continuity authority to this relocation plan and its CAS receipt."""
    from polylogue.storage.sqlite.migration_runner import DurableChangeTrain

    assert isinstance(train, DurableChangeTrain)
    refresh_refs = tuple(
        ref.removeprefix("proof:source-continuity-refresh:")
        for ref in train.proof_refs
        if ref.startswith("proof:source-continuity-refresh:")
    )
    if refresh_refs != item.continuity_receipt_digests:
        raise ArchiveRootRelocationError("archive-root relocation exact refresh proof changed")
    if before:
        return
    if relocation_receipt is None or relocation_receipt.plan_sha256 != plan.plan_sha256:
        raise ArchiveRootRelocationError("archive-root relocation exact receipt binding is missing")
    receipt_digest = relocation_receipt.prepared_receipt_sha256 or relocation_receipt.receipt_sha256
    if f"proof:archive-root-relocation:{receipt_digest}" not in train.proof_refs:
        raise ArchiveRootRelocationError("archive-root relocation exact receipt binding is missing")
    if train.source_continuity_evidence is None:
        return
    transition_refs = tuple(
        ref.removeprefix("proof:source-continuity-relocation:")
        for ref in train.proof_refs
        if ref.startswith("proof:source-continuity-relocation:")
    )
    matches = 0
    try:
        with existing_maintenance_receipt_directory(root, "source-continuity-relocations") as directory_fd:
            for digest in transition_refs:
                path = root / ".maintenance-state" / "source-continuity-relocations" / f"{digest}.json"
                encoded = None if directory_fd is None else read_optional_receipt(directory_fd, path.name)
                if encoded is None:
                    raise ArchiveRootRelocationError("archive-root relocation exact transition proof is missing")
                payload = json.loads(encoded)
                if not isinstance(payload, dict) or payload.pop("transition_sha256", None) != digest:
                    raise ArchiveRootRelocationError("archive-root relocation exact transition proof changed")
                if _canonical_sha256(payload) != digest:
                    raise ArchiveRootRelocationError("archive-root relocation exact transition proof changed")
                if (
                    payload.get("relocation_plan_sha256") == plan.plan_sha256
                    and payload.get("relocation_receipt_sha256") == receipt_digest
                ):
                    matches += 1
    except (MaintenanceReceiptPathError, json.JSONDecodeError) as exc:
        raise ArchiveRootRelocationError("archive-root relocation exact transition proof is unreadable") from exc
    if matches != 1:
        raise ArchiveRootRelocationError("archive-root relocation exact transition proof is missing")


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
            backup_configured_root=Path(plan.new_configured_root),
            backup_archive_root=root,
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
        plan.backup_root_device,
        plan.backup_root_inode,
    ):
        raise ArchiveRootRelocationError("archive-root relocation moved-root identity authority changed")
    backup_tiers = {item.tier: (item.backup_device, item.backup_inode) for item in plan.tiers}
    if len(plan.tiers) != len(ArchiveTier) or set(backup_tiers) != {tier.value for tier in ArchiveTier}:
        raise ArchiveRootRelocationError("archive-root relocation plan tier evidence is incomplete")
    pending_receipt = _load_receipt_for_update(_receipt_path(root, plan))
    prepared_publication = _has_matching_prepared_publication_receipt(plan, pending_receipt)
    prepared_resume = prepared_publication or _has_matching_initial_preparation_receipt(plan, pending_receipt)
    if plan.format == _LEGACY_PLAN_FORMAT and not prepared_resume:
        raise ArchiveRootRelocationError(
            "archive-root relocation v3 plan lacks sealed leaf identities before publication; create a v4 plan"
        )
    snapshots = tuple(
        _tier_snapshot(
            root,
            tier,
            backup_device=backup_tiers[tier.value][0],
            backup_inode=backup_tiers[tier.value][1],
            active_index_pointer=plan.active_index_pointer,
        )
        for tier in ArchiveTier
    )
    if snapshots != plan.tiers:
        raise ArchiveRootRelocationError("archive-root relocation tier evidence changed")
    _check_backup_against_live(root, manifest=manifest, receipt=receipt, snapshots=snapshots)
    _validate_active_index_pointer(root, plan.active_index_pointer)
    _validate_index_generation_state(
        root,
        plan.index_generations,
        plan.active_index_pointer,
        allow_post_publication=prepared_publication,
    )
    allowed_pending_relocation_receipt_sha256 = (
        (pending_receipt.prepared_receipt_sha256 or pending_receipt.receipt_sha256)
        if pending_receipt is not None and pending_receipt.state == "prepared"
        else None
    )
    if (
        pending_receipt is not None
        and pending_receipt.manifest_after_sha256
        and len(pending_receipt.manifest_after_sha256) != len(plan.durable_trains)
    ):
        raise ArchiveRootRelocationError("archive-root relocation receipt manifest binding changed")
    for index, item in enumerate(plan.durable_trains):
        path = Path(item.path)
        train = load_durable_change_train_manifest(path)
        manifest_sha256 = _sha256_file(path)
        continuity_refs = tuple(
            ref.removeprefix("proof:source-continuity-refresh:")
            for ref in train.proof_refs
            if ref.startswith("proof:source-continuity-refresh:")
        )
        before = manifest_sha256 == item.before_manifest_sha256
        after = (
            pending_receipt is not None
            and bool(pending_receipt.manifest_after_sha256)
            and manifest_sha256 == pending_receipt.manifest_after_sha256[index]
        )
        if not before and not after:
            raise ArchiveRootRelocationError(f"archive-root relocation manifest changed: {path}")
        try:
            _validate_archive_root_relocation_receipts(
                root,
                train,
                allowed_pending_relocation_receipt_sha256=allowed_pending_relocation_receipt_sha256,
            )
        except DurableChangeTrainError as exc:
            raise ArchiveRootRelocationError(
                f"archive-root relocation retained train authority is invalid: {path}"
            ) from exc
        if before and continuity_refs != item.continuity_receipt_digests:
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
        _validate_plan_continuity_binding(
            root,
            plan=plan,
            item=item,
            train=train,
            before=before,
            relocation_receipt=pending_receipt,
        )


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
    """CAS-rewrite released durable manifests under the owned offline boundary."""
    _verify_plan(plan)
    if authorization != plan.plan_sha256 or plan.bound_confirmation != "archive-root-relocation":
        raise ArchiveRootRelocationError("archive-root relocation authorization does not bind this plan")
    if str(root.absolute()) != plan.new_configured_root or str(root) != plan.new_resolved_root:
        raise ArchiveRootRelocationError("archive-root relocation plan is bound to a different configured root")
    _revalidate_plan_live_state(root, plan)
    receipt_path = _receipt_path(root, plan)
    retained_plan_path = _retain_plan(root, plan)
    command = (
        f"POLYLOGUE_ARCHIVE_ROOT={shlex.quote(plan.new_configured_root)} polylogue ops maintenance "
        f"archive-root-relocation apply --plan {shlex.quote(str(retained_plan_path))} "
        f"--authorize {plan.plan_sha256} --output-format json"
    )
    before_hashes = tuple(item.before_manifest_sha256 for item in plan.durable_trains)
    pointer_fields = _pointer_receipt_fields(plan.active_index_pointer)
    receipt = _sealed_receipt(
        state="prepared",
        revision=0,
        plan_sha256=plan.plan_sha256,
        authorization=authorization,
        manifest_before_sha256=before_hashes,
        manifest_after_sha256=(),
        active_index_pointer_old_target=pointer_fields[0],
        active_index_pointer_new_target=pointer_fields[1],
        active_index_pointer_new_resolved_target=pointer_fields[2],
        resume_command=command,
    )
    existing_receipt = _load_receipt_for_update(receipt_path)
    if existing_receipt is not None:
        receipt = existing_receipt
        if receipt.plan_sha256 != plan.plan_sha256 or receipt.authorization != authorization:
            raise ArchiveRootRelocationError("archive-root relocation receipt belongs to another plan")
        if not _receipt_pointer_fields_match(receipt, pointer_fields):
            raise ArchiveRootRelocationError("archive-root relocation receipt active index pointer binding changed")
        if receipt.state == "committed":
            if tuple(_sha256_file(Path(item.path)) for item in plan.durable_trains) != receipt.manifest_after_sha256:
                raise ArchiveRootRelocationError("archive-root relocation committed receipt does not match manifests")
            return ArchiveRootRelocationResult(
                state="committed",
                plan_sha256=plan.plan_sha256,
                receipt_path=str(receipt_path),
                changed_manifests=tuple(item.path for item in plan.durable_trains if _requires_train_update(item)),
            )
    else:
        _write_receipt(receipt_path, receipt, expected=None)
    preparation_receipt_sha256 = receipt.prepared_receipt_sha256 or receipt.receipt_sha256
    if receipt.state == "prepared" and not receipt.manifest_after_sha256:
        expected_after: list[str] = []
        for item in plan.durable_trains:
            path = Path(item.path)
            train = load_durable_change_train_manifest(path)
            if _sha256_file(path) != item.before_manifest_sha256:
                raise ArchiveRootRelocationError(
                    f"archive-root relocation manifest changed before expected CAS binding: {path}"
                )
            expected_train = (
                _relocated_train(
                    root,
                    plan=plan,
                    item=item,
                    train=train,
                    relocation_receipt_sha256=preparation_receipt_sha256,
                )
                if _requires_train_update(item)
                else train
            )
            expected_after.append(_train_manifest_sha256(expected_train))
        bound_prepared = _sealed_receipt(
            state="prepared",
            revision=1,
            plan_sha256=plan.plan_sha256,
            authorization=authorization,
            manifest_before_sha256=before_hashes,
            manifest_after_sha256=tuple(expected_after),
            active_index_pointer_old_target=pointer_fields[0],
            active_index_pointer_new_target=pointer_fields[1],
            active_index_pointer_new_resolved_target=pointer_fields[2],
            resume_command=command,
            prepared_receipt_sha256=preparation_receipt_sha256,
        )
        _write_receipt(receipt_path, bound_prepared, expected=receipt.receipt_sha256)
        receipt = bound_prepared
    _publish_index_generation_state(root, plan.index_generations, plan.active_index_pointer)
    _publish_active_index_pointer(root, plan.active_index_pointer)
    after_hashes: list[str] = []
    for index, item in enumerate(plan.durable_trains):
        path = Path(item.path)
        train = load_durable_change_train_manifest(path)
        actual_hash = _sha256_file(path)
        if actual_hash == item.before_manifest_sha256 and _requires_train_update(item):
            updated = _relocated_train(
                root,
                plan=plan,
                item=item,
                train=train,
                relocation_receipt_sha256=preparation_receipt_sha256,
            )
            if _train_manifest_sha256(updated) != receipt.manifest_after_sha256[index]:
                raise ArchiveRootRelocationError("archive-root relocation expected manifest binding changed")
            write_durable_change_train_manifest(path, updated, expected_revision=item.before_revision)
        elif actual_hash != receipt.manifest_after_sha256[index]:
            raise ArchiveRootRelocationError(
                f"archive-root relocation manifest is neither exact before nor after: {path}"
            )
        after_hashes.append(_sha256_file(path))
    committed = _sealed_receipt(
        state="committed",
        revision=2,
        plan_sha256=plan.plan_sha256,
        authorization=authorization,
        manifest_before_sha256=before_hashes,
        manifest_after_sha256=tuple(after_hashes),
        active_index_pointer_old_target=pointer_fields[0],
        active_index_pointer_new_target=pointer_fields[1],
        active_index_pointer_new_resolved_target=pointer_fields[2],
        resume_command=command,
        prepared_receipt_sha256=preparation_receipt_sha256,
    )
    _write_receipt(receipt_path, committed, expected=receipt.receipt_sha256)
    return ArchiveRootRelocationResult(
        state="committed",
        plan_sha256=plan.plan_sha256,
        receipt_path=str(receipt_path),
        changed_manifests=tuple(item.path for item in plan.durable_trains if _requires_train_update(item)),
    )

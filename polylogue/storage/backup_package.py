"""Backup and portability operations for the Polylogue archive.

Provides a local-first backup command for tiered archives.
Backups copy the authority and precious tiers plus referenced blobs: audit.db,
source.db, user.db, embeddings.db, and blob files. Rebuildable index.db and
disposable ops.db are omitted by profiles that do not request full evidence.

Each SQLite tier is copied through a pinned read transaction and SQLite's
backup API. The package binds the copied image separately from the original
physical tier, whose fingerprint remains authority for migration admission.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager, closing, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Literal

from pydantic import BaseModel

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.content_identity import ContentIdentityRefusal, payload_content_identity
from polylogue.core.durable_fs import atomic_replace, sync_directory, sync_tree
from polylogue.core.errors import SchemaSkew
from polylogue.core.write_lease import require_write_lease
from polylogue.paths import archive_root
from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation, assert_owns_archive_location
from polylogue.storage.backup_attestation import (
    VERIFICATION_RECEIPT_FORMAT,
    archive_tier_paths,
    assert_archive_format_authority,
    sign_verification_receipt,
)
from polylogue.storage.backup_blob_closure import (
    SOURCE_DECLARED_ABSENT_FILE,
    package_blob_closure,
    read_source_declared_absent_assertion,
    source_blob_reservations,
)
from polylogue.storage.blob_integrity import (
    BlobLivenessProjection,
    BlobReferenceDebtReport,
    _raw_session_reference_rows,
    blob_reference_debt_from_projection,
    project_source_blob_liveness,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.source_blob_restoration import (
    RetainedBlobSource,
    RetainedBlobSourceKind,
    is_legacy_append_without_window,
    read_prior_full_source_receipts,
    read_raw_source_evidence,
    retained_blob_source_candidates,
    retained_source_location,
    source_window_holds_blob,
    stage_exact_blob,
)
from polylogue.storage.source_zip_replay import MemberCandidate, MemberCandidateCache, zip_reacquired_unit
from polylogue.storage.sqlite.connection_profile import (
    NativeSQLCustodyOwner,
    _close_failed_native_construction,
    open_readonly_connection,
    open_scratch_connection,
)

BackupProfile = Literal["full_evidence", "user_overlays", "rebuildable_cache_exclude", "diagnostics_bundle"]
BACKUP_PROFILES: tuple[BackupProfile, ...] = (
    "full_evidence",
    "user_overlays",
    "rebuildable_cache_exclude",
    "diagnostics_bundle",
)
_MISSING_BLOB_WARNING_SAMPLE_LIMIT = 10
_VERIFICATION_RECEIPT_FILE = "verification-receipt.json"
_BLOB_REFERENCE_EVIDENCE_FILE = "blob-reference-evidence.json"
_ARCHIVE_AUTHORITY_FILES = (
    ".polylogue-format.json",
    ".maintenance-state/durable-change-trains/.bootstrap",
    ".maintenance-state/durable-change-trains/.bootstrap.pending",
)


_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_RECOVERY_PROOF_KINDS = frozenset(
    {
        "direct_file_sha256",
        "historical_snapshot_prefix_sha256",
        "zip_reacquired_payload",
        "live_append_segment_sha256",
        "historical_append_segment_sha256",
    }
)
_RECOVERABILITY_FAILURE_KINDS = frozenset(
    {
        "no_replay_candidate",
        "source_missing",
        "legacy_append_window_missing",
        "acquisition_coordinate",
        "replay_error",
        "container_member_rejected",
        "inexact_payload",
        "hash_mismatch",
    }
)


def _archive_authority_file_names(root: Path, included_tiers: Iterable[str]) -> tuple[str, ...]:
    """Name original birth and numbered history evidence in this snapshot.

    These copied receipts retain their original physical archive bindings;
    they are not executable authority for the backup's new SQLite inodes.
    Lock files are current process custody, not durable train evidence.
    """
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.durable_change_train import durable_train_manifest_paths

    manifest_root = root / ".maintenance-state" / "durable-change-trains"
    included = set(included_tiers)
    return _ARCHIVE_AUTHORITY_FILES + tuple(
        str(path.relative_to(root))
        for tier in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT)
        if tier.value in included
        for path in durable_train_manifest_paths(manifest_root, tier)
    )


class BackupResult(BaseModel):
    """Result of a backup operation."""

    ok: bool
    output_path: str | None = None
    backup_mode: str = "archive_file_set"
    backup_profile: str = "rebuildable_cache_exclude"
    db_size_bytes: int = 0
    blob_count: int = 0
    blob_size_bytes: int = 0
    elapsed_s: float = 0.0
    error: str | None = None
    check_only: bool = False
    warnings: list[str] = []
    backed_up_files: list[str] = []
    omitted_tiers: list[str] = []
    verified: bool = False
    verification: dict[str, object] = {}


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_real_backup_directory(path: Path, *, label: str) -> Path:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} is missing: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{label} is not a real directory: {path}")
    return path.resolve(strict=True)


def _require_regular_backup_artifact(path: Path, *, backup_root: Path, label: str) -> os.stat_result:
    root_resolved = _require_real_backup_directory(backup_root, label="backup root")
    try:
        relative = path.relative_to(backup_root)
    except ValueError as exc:
        raise RuntimeError(f"{label} is outside the backup root: {path}") from exc
    current = backup_root
    for part in relative.parts[:-1]:
        current /= part
        _require_real_backup_directory(current, label=f"{label} parent")
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} is not a real regular file: {path}")
    if metadata.st_nlink != 1:
        raise RuntimeError(f"{label} has multiple hard links: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise RuntimeError(f"{label} resolves outside the backup root: {path}")
    return metadata


def _regular_backup_blob_files(backup_root: Path) -> list[Path]:
    blob_root = backup_root / "blob"
    if not blob_root.exists() and not blob_root.is_symlink():
        return []
    _require_real_backup_directory(blob_root, label="backup blob root")
    files: list[Path] = []
    for candidate in sorted(blob_root.rglob("*")):
        metadata = candidate.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            _require_real_backup_directory(candidate, label="backup blob directory")
            continue
        _require_regular_backup_artifact(candidate, backup_root=backup_root, label="backup blob")
        files.append(candidate)
    return files


def _reject_sqlite_sidecars(path: Path) -> None:
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
        sidecar = Path(f"{path}{suffix}")
        if sidecar.exists() or sidecar.is_symlink():
            raise RuntimeError(f"backup tier has an unbound SQLite sidecar: {sidecar}")


def _sqlite_sidecar_paths(root: Path) -> frozenset[Path]:
    return frozenset(
        candidate
        for candidate in root.rglob("*")
        if candidate.name.endswith(_SQLITE_SIDECAR_SUFFIXES) and not stat.S_ISDIR(candidate.lstat().st_mode)
    )


def _discard_scratch_verification_sidecars(scratch_root: Path, *, copied: frozenset[Path]) -> list[Path]:
    """Drop the WAL/SHM files verification's own reads materialized.

    A tier copied out of a WAL-mode archive still declares WAL journalling in
    its header even though the backup carries no ``-wal``: ``_backup_sqlite``
    closes the SQLite backup image before publishing it. The archive-format
    lineage gate then inspects that copy through a ``mode=ro`` connection,
    and SQLite materializes an empty ``-shm``/``-wal`` pair for it. Those
    bytes are the verifier's own side effect, not backup content, so leaving
    them would make the scratch inventory disagree with the published backup
    and refuse a valid backup. Sidecars that arrived with the copy are kept so
    the published-sidecar refusal still fires on them.
    """
    created = sorted(_sqlite_sidecar_paths(scratch_root) - copied)
    for sidecar in created:
        sidecar.unlink(missing_ok=True)
    return created


def _backup_artifact_inventory(
    backup_root: Path,
    *,
    verified_file_hashes: Mapping[str, tuple[int, str]] | None = None,
) -> list[dict[str, object]]:
    _require_real_backup_directory(backup_root, label="backup root")
    rows: list[dict[str, object]] = []
    for candidate in sorted(backup_root.rglob("*")):
        relative = candidate.relative_to(backup_root)
        if relative == Path(_VERIFICATION_RECEIPT_FILE):
            continue
        metadata = candidate.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            _require_real_backup_directory(candidate, label="backup artifact directory")
            rows.append({"path": str(relative), "type": "directory"})
            continue
        if candidate.name.endswith(_SQLITE_SIDECAR_SUFFIXES):
            raise RuntimeError(f"backup contains an unbound SQLite sidecar: {candidate}")
        _require_regular_backup_artifact(candidate, backup_root=backup_root, label="backup artifact")
        verified = (verified_file_hashes or {}).get(str(relative))
        if verified is not None and verified[0] != metadata.st_size:
            raise RuntimeError(f"verified backup artifact changed while receipt evidence was built: {candidate}")
        rows.append(
            {
                "path": str(relative),
                "type": "file",
                "size_bytes": metadata.st_size,
                "sha256": verified[1] if verified is not None else _sha256_file(candidate),
            }
        )
    return rows


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _open_backup_readonly_connection(
    path: Path,
    *,
    immutable: bool,
    timeout_class: str,
) -> sqlite3.Connection:
    """Open backup evidence recorded at or below this runtime's tier version.

    A backup-gated migration is authorized by a backup taken before it runs, so
    that backup carries the tier's pre-migration version by construction.
    Demanding the current version here would make every such migration
    unapplicable. A version above the expected one is still refused: this
    runtime cannot interpret it. An unstamped tier (version 0) is refused too.
    """
    # This is acquisition of SQLite evidence, not a read-model admission.
    # Stale derived identity remains evidence to retain; ordinary product
    # readers still enforce their current identity before serving rows.
    from polylogue.storage.io_phase_metrics import connection_cursor
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_BASELINE_VERSION_BY_TIER, ARCHIVE_VERSION_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner, _close_failed_native_construction

    connection = open_readonly_connection(path, immutable=immutable, timeout_class=timeout_class, validate_schema=False)
    owner = NativeSQLCustodyOwner(connection)
    try:
        try:
            tier = ArchiveTier(path.stem)
        except ValueError:
            tier = None
        if tier is not None:
            with connection_cursor(owner.require_connection(), "PRAGMA user_version") as cursor:
                version = int(cursor.fetchone()[0])
            if not ARCHIVE_BASELINE_VERSION_BY_TIER[tier] <= version <= ARCHIVE_VERSION_BY_TIER[tier]:
                raise SchemaSkew(tier.value, ARCHIVE_VERSION_BY_TIER[tier], version)
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    return owner.handoff()


def _sqlite_user_version(path: Path) -> int:
    # A live WAL-mode tier can hold committed frames its main file does not;
    # an immutable open refuses such a file, so it is read through the WAL.
    # Package artifacts and closed backup images carry no frames and stay
    # immutable, which creates no sidecar beside them.
    wal = path.with_name(f"{path.name}-wal")
    try:
        has_frames = wal.stat().st_size > 0
    except FileNotFoundError:
        has_frames = False
    with closing(
        _open_backup_readonly_connection(path, immutable=not has_frames, timeout_class="offline-bulk")
    ) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)


def _readable_sqlite_index(path: Path) -> bool:
    """Return whether an active-pointer candidate is a readable SQLite index.

    Backup follows a genuinely live external index target, but a malformed or
    stale pointer must not turn an otherwise valid backup into a copy of
    arbitrary bytes. Relocation still authenticates that pointer separately.
    """
    # Non-SQLite pointer targets are not archive operands. Once the literal
    # header selects a SQLite operand, read faults must reach the caller.
    with path.open("rb") as stream:
        if stream.read(16) != b"SQLite format 3\x00":
            return False
    _sqlite_user_version(path)
    return True


def _sqlite_source_fingerprint(path: Path, *, snapshot_path: Path, user_version: int) -> dict[str, object]:
    from polylogue.storage.sqlite.physical_file import physical_file_sha256

    metadata = path.stat()
    physical = physical_file_sha256(path, expected_device=metadata.st_dev, expected_inode=metadata.st_ino)
    wal_path = path.with_name(f"{path.name}-wal")
    try:
        wal_metadata = wal_path.stat()
    except FileNotFoundError:
        wal_metadata = None
    wal = None
    if wal_metadata is not None and wal_metadata.st_size:
        wal_digest = physical_file_sha256(
            wal_path, expected_device=wal_metadata.st_dev, expected_inode=wal_metadata.st_ino
        )
        wal = {"size_bytes": wal_digest.size_bytes, "sha256": wal_digest.sha256}
    return {
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "size_bytes": physical.size_bytes,
        "sha256": physical.sha256,
        "user_version": user_version,
        "wal": wal,
        "snapshot": {
            "size_bytes": snapshot_path.stat().st_size,
            "sha256": _sha256_file(snapshot_path),
            "user_version": user_version,
        },
    }


def _archive_root_source_identity(root: Path) -> dict[str, object]:
    """Capture the pre-move directory identity later authenticated by the receipt."""
    configured = root.absolute()
    resolved = root.resolve(strict=True)
    metadata = resolved.stat()
    return {
        "configured_path": str(configured),
        "resolved_path": str(resolved),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }


def _json_str_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _all_archive_tiers(root: Path) -> dict[str, Path]:
    tiers = archive_tier_paths(root)
    pointer = root / ".index-active-pointer"
    try:
        pointer_metadata = pointer.lstat()
    except OSError:
        return tiers
    try:
        if stat.S_ISLNK(pointer_metadata.st_mode):
            raw_target = os.readlink(pointer)
        elif stat.S_ISREG(pointer_metadata.st_mode) and pointer_metadata.st_nlink == 1:
            raw_target = pointer.read_text(encoding="utf-8").strip()
        else:
            return tiers
    except (OSError, UnicodeDecodeError):
        return tiers
    configured_target = Path(raw_target)
    if not configured_target.is_absolute() or configured_target.name != "index.db":
        return tiers
    if (
        not configured_target.is_relative_to(root.absolute())
        and configured_target.is_file()
        and not configured_target.is_symlink()
        and _readable_sqlite_index(configured_target)
    ):
        tiers["index"] = configured_target
        return tiers
    if (
        configured_target.is_relative_to(root.absolute())
        and configured_target.is_file()
        and configured_target.resolve().is_relative_to(root.resolve())
    ):
        tiers["index"] = configured_target
        return tiers

    # An inode-preserving root move leaves the absolute pointer and the
    # promoted conventional symlink carrying the retired root until the
    # relocation operation publishes their mapped forms. Locate the unique
    # conventional symlink paired with that pointer, including a canonical
    # index below the archive root rather than assuming ``root/index.db``.
    mapped_candidates: list[tuple[int, Path]] = []
    target_parts = configured_target.relative_to(configured_target.anchor).parts
    conventional_candidates = tuple(root.joinpath(*target_parts[-depth:]) for depth in range(1, len(target_parts) + 1))
    for conventional in dict.fromkeys(conventional_candidates):
        relative_conventional = conventional.relative_to(root)
        if ".index-generations" in relative_conventional.parts:
            continue
        relative_parts = relative_conventional.parts
        if len(relative_parts) > len(configured_target.parts) or configured_target.parts[-len(relative_parts) :] != (
            relative_parts
        ):
            continue
        if conventional.is_file() and not conventional.is_symlink():
            mapped_candidates.append((len(relative_parts), conventional))
            continue
        if conventional.is_symlink():
            target = Path(os.readlink(conventional))
            if not target.is_absolute():
                continue
            try:
                relative = target.relative_to(configured_target.parent)
            except ValueError:
                continue
            mapped = conventional.parent / relative
            if (
                len(relative.parts) >= 3
                and relative.parts[0] == ".index-generations"
                and relative.parts[-1] == "index.db"
                and mapped.is_file()
                and not mapped.is_symlink()
                and _readable_sqlite_index(mapped)
            ):
                mapped_candidates.append((len(relative_parts), mapped))
    longest_suffix = max((length for length, _path in mapped_candidates), default=0)
    unique_candidates = tuple(dict.fromkeys(path for length, path in mapped_candidates if length == longest_suffix))
    if len(unique_candidates) == 1:
        tiers["index"] = unique_candidates[0]
    return tiers


def _profile_archive_tiers(root: Path, profile: BackupProfile) -> dict[str, Path]:
    all_tiers = _all_archive_tiers(root)
    if profile == "full_evidence":
        return all_tiers
    if profile == "user_overlays":
        return {"user": all_tiers["user"], "audit": all_tiers["audit"]}
    if profile == "diagnostics_bundle":
        return {"ops": all_tiers["ops"]}
    return {
        "source": all_tiers["source"],
        "user": all_tiers["user"],
        "embeddings": all_tiers["embeddings"],
        "audit": all_tiers["audit"],
    }


def _optional_profile_tiers(profile: BackupProfile) -> set[str]:
    if profile == "full_evidence":
        return {"ops", "audit"}
    if profile == "rebuildable_cache_exclude":
        return {"embeddings", "audit"}
    if profile == "user_overlays":
        return {"audit"}
    if profile == "diagnostics_bundle":
        return {"audit"}
    return set()


def _archive_layout_present(root: Path) -> bool:
    return any(path.exists() for path in _all_archive_tiers(root).values())


def _require_readable_sqlite(path: Path) -> None:
    """Preserve an unreadable tier as the original failure, never a diagnostic value."""
    from polylogue.core.sql_settlement import current_native_sql_lifetimes
    from polylogue.storage.io_phase_metrics import connection_cursor
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner, _close_failed_native_construction

    connection = _open_backup_readonly_connection(path, immutable=False, timeout_class="background-read")
    owner = NativeSQLCustodyOwner(connection, lifetime_dependencies=current_native_sql_lifetimes())
    try:
        with connection_cursor(owner.require_connection(), "SELECT 1 FROM sqlite_master LIMIT 1"):
            pass
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    else:
        owner.close()


def _check_prerequisites(
    *, profile: BackupProfile = "rebuildable_cache_exclude", archive_root_path: Path | None = None
) -> list[str]:
    """Return a list of warning/error strings for backup prerequisites."""
    warnings: list[str] = []

    root = archive_root_path or archive_root()
    if not _archive_layout_present(root):
        return [f"archive tiers not found under {root}"]

    optional_tiers = _optional_profile_tiers(profile)
    for tier, path in _profile_archive_tiers(root, profile).items():
        if not path.exists():
            if tier in optional_tiers:
                continue
            warnings.append(f"{tier}.db not found at {path}")
            continue
        _require_readable_sqlite(path)

    # Allow for the backup copy plus a scratch restore during verification.
    try:
        db_size = 0
        for path in _profile_archive_tiers(root, profile).values():
            if path.exists():
                db_size += path.stat().st_size
                wal = path.with_suffix(".db-wal")
                if wal.exists():
                    db_size += wal.stat().st_size
        # Leave headroom beyond the two simultaneous file sets.
        needed = int(db_size * 2.5)
        import os

        st = os.statvfs(str(root))
        free = st.f_frsize * st.f_bavail
        if free < needed:
            warnings.append(
                f"low disk space: {free / (1024**3):.1f} GB free, "
                f"~{needed / (1024**3):.1f} GB needed for backup and scratch verification"
            )
    except Exception as exc:
        # A swallowed failure here previously left the disk-space check
        # unrepresented in `warnings` at all — indistinguishable from "disk
        # space is fine". Surface it as its own warning so the check's own
        # failure is visible (polylogue-cpf.4).
        warnings.append(f"disk space check failed: {exc}")

    return warnings


def _has_backup_error(warnings: list[str]) -> bool:
    return any("not found" in warning for warning in warnings)


def _backup_sqlite(src: Path, dst: Path, *, archive_root_path: Path) -> tuple[int, dict[str, object]]:
    """Copy the selected read cut without checkpointing or excluding readers.

    The outer archive writer custody binds sibling tiers and retained blobs.
    BEGIN plus the first schema read pins this tier before backup starts, so
    a later WAL commit cannot restart the copy on a newer snapshot.
    """
    require_write_lease("archive backup snapshot", archive_root=archive_root_path)
    check_compute_cancelled()
    live_path = src.resolve(strict=True)
    conn = _open_backup_readonly_connection(live_path, immutable=False, timeout_class="offline-bulk")
    source_owner = NativeSQLCustodyOwner(conn, lifetime_dependencies=(dst,))
    try:
        from polylogue.storage.io_phase_metrics import connection_cursor

        conn.execute("BEGIN").close()
        with connection_cursor(conn, "SELECT 1 FROM sqlite_master LIMIT 1") as cursor:
            cursor.fetchall()
        with connection_cursor(conn, "PRAGMA user_version") as cursor:
            user_version = int(cursor.fetchone()[0])
        with connection_cursor(conn, "PRAGMA data_version") as cursor:
            selected_version = int(cursor.fetchone()[0])
        # The image is private while SQLite populates it, then retains the
        # original tier's metadata after physical destination settlement.
        dst.touch(mode=0o600, exist_ok=False)
        destination_owner = open_scratch_connection(dst, lifetime_dependencies=(dst,))
        try:
            # Page batches bound cooperative cancellation, not accepted input.
            conn.backup(
                destination_owner.require_connection(),
                pages=256,
                progress=lambda _status, _remaining, _total: check_compute_cancelled(),
            )
            check_compute_cancelled()
        except BaseException as primary:
            _close_failed_native_construction(destination_owner, primary)
            raise
        else:
            destination_owner.close()
        shutil.copystat(live_path, dst)
        conn.rollback()
        with connection_cursor(conn, "PRAGMA data_version") as cursor:
            before_fingerprint = int(cursor.fetchone()[0])
        fingerprint = _sqlite_source_fingerprint(live_path, snapshot_path=dst, user_version=user_version)
        with connection_cursor(conn, "PRAGMA data_version") as cursor:
            after_fingerprint = int(cursor.fetchone()[0])
        # A changing live tier is still a valid recovery image. It cannot
        # authorize a later migration against physical bytes newer than its cut.
        fingerprint["live_cut_stable"] = selected_version == before_fingerprint == after_fingerprint
    except BaseException as primary:
        _close_failed_native_construction(source_owner, primary)
        raise
    else:
        source_owner.close()
    return dst.stat().st_size, fingerprint


def _source_blob_liveness_projection(
    source_db: Path, *, index_db: Path | None = None
) -> tuple[BlobLivenessProjection, set[str]]:
    """Read complete source evidence or refuse the backup before copying blobs.

    The copied ``source.db`` keeps every source generation's rows, so the blob
    set is every generation's live bytes. Narrowing it to the newest sealed
    generation left earlier generations' rows restored without their blobs,
    and the debt report, computed from the same narrowed set, could not see
    the gap.
    """

    projection = project_source_blob_liveness(source_db, index_db=index_db, immutable=True)
    return projection, source_blob_reservations(source_db)


def _copy_source_declared_absent_assertion(
    source_db: Path, backup_root: Path, *, fingerprint: dict[str, object]
) -> Path | None:
    """Copy the optional durable source assertion into a backup package."""

    source_path = source_db.with_name(SOURCE_DECLARED_ABSENT_FILE)
    if not source_path.exists() and not source_path.is_symlink():
        return None
    _require_regular_backup_artifact(
        source_path, backup_root=source_db.parent, label="source declared-absent assertion"
    )
    if fingerprint.get("live_cut_stable") is not True or fingerprint.get("wal") is not None:
        raise RuntimeError("source declared-absent assertion does not bind the selected WAL snapshot")
    assertion, _ = read_source_declared_absent_assertion(source_path, source_db_sha256=str(fingerprint["sha256"]))
    snapshot = fingerprint.get("snapshot")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("sha256"), str):
        raise RuntimeError("source declared-absent assertion lacks the pinned image fingerprint")
    assertion["source_db_sha256"] = snapshot["sha256"]
    destination = backup_root / SOURCE_DECLARED_ABSENT_FILE
    atomic_replace(destination, json.dumps(assertion, indent=2, sort_keys=True).encode())
    return destination


def _inventory_from_liveness(projection: BlobLivenessProjection, reservations: set[str]) -> dict[str, set[str]]:
    inventory = {blob_hash: {"committed"} for blob_hash in projection.live_hashes}
    for blob_hash in reservations:
        inventory.setdefault(blob_hash, set()).add("reserved")
    return inventory


def _index_attachment_hashes(index_db: Path | None) -> set[str]:
    """Resolve attachment ownership independently of the liveness projection."""

    if index_db is None:
        return set()
    try:
        with closing(
            _open_backup_readonly_connection(index_db, immutable=True, timeout_class="offline-bulk")
        ) as index_conn:
            columns = {str(row[1]) for row in index_conn.execute("PRAGMA table_info(attachments)")}
            if "blob_hash" not in columns:
                raise RuntimeError("index.attachments is missing columns: blob_hash")
            hashes: set[str] = set()
            for (blob_hash,) in index_conn.execute(
                "SELECT DISTINCT blob_hash FROM attachments WHERE blob_hash IS NOT NULL"
            ):
                if not isinstance(blob_hash, bytes) or len(blob_hash) != 32:
                    raise RuntimeError("index.attachments has invalid blob_hash evidence")
                hashes.add(blob_hash.hex())
            return hashes
    except sqlite3.Error as exc:
        raise RuntimeError(f"index attachment ownership is unreadable: {exc}") from exc


def _blob_reference_evidence(
    projection: BlobLivenessProjection,
    *,
    index_db: Path | None,
) -> dict[str, object]:
    """Persist source resolution plus an independent index-attachment oracle.

    Blob copying follows the canonical liveness projection. The attachment
    query is deliberately separate so backup verification can contradict a
    projection that accidentally omits a readable index-only owner.
    """

    source_owners = {
        owner: sorted(hashes) for owner, hashes in projection.owner_hashes if owner.startswith("source.db.")
    }
    attachment_evidence_state = "consulted" if index_db is not None else "not_consulted"
    attachment_hashes = _index_attachment_hashes(index_db) if index_db is not None else set()
    expected_hashes = set().union(*(set(hashes) for hashes in source_owners.values()), attachment_hashes)
    omitted = expected_hashes - set(projection.live_hashes)
    if omitted:
        sample = ", ".join(sorted(omitted)[:_MISSING_BLOB_WARNING_SAMPLE_LIMIT])
        raise RuntimeError(f"canonical blob liveness projection omitted independent attachment evidence: {sample}")
    return {
        "format": "polylogue-blob-reference-evidence-v1",
        "source_owner_hashes": source_owners,
        "index_attachment_evidence": attachment_evidence_state,
        "index_attachment_hashes": sorted(attachment_hashes),
    }


def _source_recoverability_proofs(
    source_db: Path,
    *,
    root: Path,
    missing_hashes: set[str],
    unproven: list[dict[str, str]] | None = None,
    zip_payload_cache: MemberCandidateCache | None = None,
    immutable: bool = True,
    recover: Callable[[str, int, IO[bytes]], bool] | None = None,
) -> list[dict[str, str]]:
    """Prove missing source-owned bytes by replaying their acquisition payload.

    The candidate source windows of each row come from
    ``retained_blob_source_candidates``, the owner raw derivation's blob
    restoration reads too. With ``recover``, a replayed payload is a proof
    only once ``recover`` accepted it as the blob's exact bytes
    (``recover(blob_hash, size, stream)``, read from the ZIP member value or
    streamed from the direct-source window); a structural-only match is then
    recorded as unproven ``inexact_payload``, because a package cannot carry
    bytes it does not hold.
    """
    if not missing_hashes:
        return []
    zip_payload_cache = zip_payload_cache if zip_payload_cache is not None else {}
    by_hash: dict[str, list[dict[str, object]]] = {}
    prior_full_receipts: dict[str, tuple[tuple[int, int], ...]] = {}
    with closing(
        _open_backup_readonly_connection(
            source_db,
            immutable=immutable,
            timeout_class="offline-bulk" if immutable else "background-read",
        )
    ) as conn:
        from polylogue.storage.io_phase_metrics import connection_cursor
        from polylogue.storage.sqlite.archive_tiers import ARCHIVE_BASELINE_VERSION_BY_TIER, ARCHIVE_VERSION_BY_TIER
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        # The Source baseline carries captured_coordinate, the newest field
        # this recovery reader interprets. Later additive trains do not
        # invalidate the authenticated predecessor's acquisition coordinates.
        # Other versions remain opaque evidence, never guessed current coordinates.
        with connection_cursor(conn, "PRAGMA user_version") as cursor:
            version = int(cursor.fetchone()[0])
        if (
            not ARCHIVE_BASELINE_VERSION_BY_TIER[ArchiveTier.SOURCE]
            <= version
            <= ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
        ):
            raise SchemaSkew(ArchiveTier.SOURCE.value, ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE], version)
        reference_rows = _raw_session_reference_rows(conn)
        for row in reference_rows:
            blob_hash = str(row.get("blob_hash") or "")
            if blob_hash in missing_hashes:
                if is_legacy_append_without_window(row):
                    evidence = read_raw_source_evidence(conn, str(row["raw_id"]))
                    row["receipt_order"] = None if evidence is None else evidence["receipt_order"]
                by_hash.setdefault(blob_hash, []).append(row)
                if is_legacy_append_without_window(row):
                    prior_full_receipts[str(row["raw_id"])] = read_prior_full_source_receipts(conn, row)
    proofs: list[dict[str, str]] = []
    for blob_hash in sorted(missing_hashes):
        rows = by_hash.get(blob_hash, [])
        errors: list[str] = []
        for row in rows:
            source_path = row.get("source_path")
            if not isinstance(source_path, str) or not source_path:
                errors.append("no_source_path")
                continue
            resolved, is_container = retained_source_location(row, root)
            candidates = retained_blob_source_candidates(
                row,
                container_member=is_container,
                prior_full_observations=prior_full_receipts.get(str(row["raw_id"]), ()),
            )
            if not candidates:
                errors.append(
                    "legacy_append_window_missing" if is_legacy_append_without_window(row) else "no_replay_candidate"
                )
                continue
            proven: RetainedBlobSource | None = None
            for candidate in candidates:
                window = candidate.window
                if window is None:
                    # A container unit is proven by its digests and, when the
                    # package must carry it, streamed from its member again:
                    # a preserved member can be gigabytes.
                    unit, error = zip_reacquired_unit(
                        row,
                        source_path=resolved,
                        zip_payload_cache=zip_payload_cache,
                    )
                    matched = error is None and unit is not None and _unit_matches_reference(row, unit, blob_hash)
                    if matched and recover is not None and unit is not None:
                        try:
                            if unit.open_payload is None:
                                exact = False
                            else:
                                with unit.open_payload() as unit_stream:
                                    exact = recover(blob_hash, _recorded_blob_size(row, unit.size_bytes), unit_stream)
                        except (OSError, zipfile.BadZipFile, LookupError, ContentIdentityRefusal) as exc:
                            matched, error = False, f"error:{type(exc).__name__}"
                        else:
                            if not exact:
                                matched, error = False, "inexact_payload"
                else:
                    # A direct-source window is proven by its bytes' hash and
                    # streamed, so a large window costs no memory.
                    try:
                        matched, error = source_window_holds_blob(Path(resolved), window, blob_hash=blob_hash)
                        if matched and recover is not None:
                            with Path(resolved).open("rb") as window_handle:
                                window_handle.seek(window.start)
                                if not recover(blob_hash, window.end - window.start, window_handle):
                                    matched, error = False, "inexact_payload"
                    except OSError as exc:
                        matched, error = False, f"error:{exc}"
                if matched:
                    proven = candidate
                    break
                errors.append(error or "hash_mismatch")
            if proven is not None:
                candidate = proven
                # An append proof names the window whose bytes were hashed --
                # the recorded window, the observed ``[0, end)`` prefix, or a
                # legacy append's replayed window -- so verification rebuilds
                # exactly that window, never the row's own offsets.
                window = (
                    candidate.window
                    if candidate.kind
                    in {RetainedBlobSourceKind.APPEND_WINDOW, RetainedBlobSourceKind.LEGACY_APPEND_WINDOW}
                    else None
                )
                proofs.append(
                    {
                        "blob_hash": blob_hash,
                        "blob_size": str(row.get("size_bytes")) if row.get("size_bytes") is not None else "",
                        "kind": candidate.kind.value,
                        "source_path": resolved,
                        "raw_id": str(row.get("ref_id") or ""),
                        "source_index": str(row.get("source_index")) if row.get("source_index") is not None else "",
                        "origin": str(row.get("origin") or ""),
                        "capture_mode": str(row.get("capture_mode") or ""),
                        "coordinate_format": str(row.get("coordinate_format") or ""),
                        "entry_ordinal": str(row.get("entry_ordinal")) if row.get("entry_ordinal") is not None else "",
                        "split_index": str(row.get("split_index")) if row.get("split_index") is not None else "",
                        "addressing_mode": str(row.get("addressing_mode") or ""),
                        "content_identity": str(row.get("content_identity") or ""),
                        "revision_kind": str(row.get("revision_kind") or ""),
                        "append_start_offset": (
                            str(window.start)
                            if window is not None
                            else str(row.get("append_start_offset"))
                            if row.get("append_start_offset") is not None
                            else ""
                        ),
                        "append_end_offset": (
                            str(window.end)
                            if window is not None
                            else str(row.get("append_end_offset"))
                            if row.get("append_end_offset") is not None
                            else ""
                        ),
                    }
                )
                break
        else:
            if unproven is not None:
                first_row = rows[0] if rows else {}
                failure_kinds = {_recoverability_failure_kind(error) for error in errors}
                failure_kind = _recoverability_failure_kind_for_attempts(failure_kinds)
                unproven.append(
                    {
                        "blob_hash": blob_hash,
                        "kind": failure_kind,
                        "reason": "; ".join(sorted(set(errors))) if errors else "no_raw_session_reference",
                        "raw_id": str(first_row.get("ref_id") or ""),
                        "source_path": str(first_row.get("source_path") or ""),
                    }
                )
    return proofs


def _recorded_blob_size(row: Mapping[str, object], fallback_size: int) -> int:
    """The retained size a recovered payload must have; the replayed unit's own when unrecorded."""
    value = row.get("size_bytes")
    if isinstance(value, (int, str)):
        try:
            return int(value)
        except ValueError:
            pass
    return fallback_size


def _recorded_content_identity(row: Mapping[str, object]) -> str | None:
    """The row's structural identity, when it records a valid one."""
    content_identity = row.get("content_identity")
    if not isinstance(content_identity, str) or len(content_identity) != 64:
        return None
    try:
        bytes.fromhex(content_identity)
    except ValueError:
        return None
    return content_identity.lower()


def _unit_matches_reference(row: Mapping[str, object], unit: MemberCandidate, blob_hash: str) -> bool:
    """Verify a replayed container unit by the recorded identity rule, from its digests."""
    content_identity = _recorded_content_identity(row)
    if content_identity is not None:
        return unit.content_identity == content_identity
    return unit.byte_identity == blob_hash


def _payload_matches_reference(row: Mapping[str, object], payload: bytes, blob_hash: str) -> bool:
    """Verify a replayed payload against its durable byte or value identity.

    Container-member rows carry a structural identity because a provider may
    reorder keys or normalize integral numbers while preserving the same
    value. Direct and legacy rows have only the retained blob hash. The
    structural check is deliberately selected from the row rather than from
    the replay path, so backup evidence cannot accidentally bless a positional
    or serialization-only match.
    """
    content_identity = row.get("content_identity")
    if isinstance(content_identity, str) and len(content_identity) == 64:
        try:
            bytes.fromhex(content_identity)
        except ValueError:
            pass
        else:
            try:
                return payload_content_identity(payload) == content_identity.lower()
            except ContentIdentityRefusal:
                # Acquisition refuses such a value, so it is not the recorded one.
                return False
    return hashlib.sha256(payload).hexdigest() == blob_hash


def _recoverability_failure_kind(error: str) -> str:
    if error == "no_raw_session_reference":
        return "no_replay_candidate"
    if error == "no_source_path":
        return "no_replay_candidate"
    if error in {"source_missing", "source_not_regular_file"}:
        return "source_missing"
    if error == "legacy_append_window_missing":
        return "legacy_append_window_missing"
    if error == "short_read":
        return "replay_error"
    if error == "member_yields_no_payload":
        return "no_replay_candidate"
    if error == "container_member_rejected":
        return "container_member_rejected"
    if error == "inexact_payload":
        return "inexact_payload"
    if error in {
        "container_coordinate_missing",
        "container_coordinate_mismatch",
        "ambiguous_container_member",
        "content_identity:unavailable",
        "replay_provider_unrecorded",
    }:
        return "acquisition_coordinate"
    if error.startswith("error:"):
        return "replay_error"
    return "hash_mismatch"


def _recoverability_failure_kind_for_attempts(kinds: set[str]) -> str:
    for kind in (
        "replay_error",
        "legacy_append_window_missing",
        "acquisition_coordinate",
        "source_missing",
        "no_replay_candidate",
        "container_member_rejected",
        "inexact_payload",
        "hash_mismatch",
    ):
        if kind in kinds:
            return kind
    return "no_replay_candidate"


def _write_blob_reference_evidence(backup_root: Path, evidence: dict[str, object]) -> Path:
    path = backup_root / _BLOB_REFERENCE_EVIDENCE_FILE
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _expected_blob_hashes_from_evidence(evidence: object) -> set[str]:
    if not isinstance(evidence, dict) or evidence.get("format") != "polylogue-blob-reference-evidence-v1":
        raise RuntimeError("backup blob reference evidence is missing or has an unknown format")
    source_owners = evidence.get("source_owner_hashes")
    attachment_hashes = evidence.get("index_attachment_hashes")
    attachment_evidence = evidence.get("index_attachment_evidence")
    if (
        not isinstance(source_owners, dict)
        or not isinstance(attachment_hashes, list)
        or attachment_evidence not in {"consulted", "not_consulted"}
    ):
        raise RuntimeError("backup blob reference evidence has invalid owner payloads")
    if attachment_evidence == "not_consulted" and attachment_hashes:
        raise RuntimeError("backup blob reference evidence records unconsulted index attachments as owners")
    if any(not isinstance(owner, str) or not isinstance(hashes, list) for owner, hashes in source_owners.items()):
        raise RuntimeError("backup blob reference evidence has invalid source owner payloads")
    values = [
        *attachment_hashes,
        *(blob_hash for hashes in source_owners.values() if isinstance(hashes, list) for blob_hash in hashes),
    ]
    if any(
        not isinstance(blob_hash, str)
        or len(blob_hash) != 64
        or any(char not in "0123456789abcdef" for char in blob_hash)
        for blob_hash in values
    ):
        raise RuntimeError("backup blob reference evidence has invalid blob hashes")
    return set(values)


def _write_blob_reference_debt_report(backup_root: Path, report: BlobReferenceDebtReport) -> Path:
    path = backup_root / "blob-reference-debt.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _copy_referenced_blobs(
    *,
    source_db: Path,
    source_blob_root: Path,
    index_db: Path | None,
    backup_root: Path,
    warnings: list[str],
) -> tuple[int, int, BlobReferenceDebtReport]:
    projection, reservations = _source_blob_liveness_projection(source_db, index_db=index_db)
    reference_evidence = _blob_reference_evidence(projection, index_db=index_db)
    inventory = _inventory_from_liveness(projection, reservations)
    hashes = set(inventory)
    store = BlobStore(source_blob_root)
    debt_report = blob_reference_debt_from_projection(
        projection,
        store=store,
        sample_size=_MISSING_BLOB_WARNING_SAMPLE_LIMIT,
    )
    missing_hashes: set[str] = set()
    if debt_report.missing_referenced_blobs:
        missing_hashes = {blob_hash for blob_hash in hashes if not store.exists(blob_hash)}
    source_owners = reference_evidence["source_owner_hashes"]
    assert isinstance(source_owners, dict)
    source_hashes = set().union(*(set(owner_hashes) for owner_hashes in source_owners.values()))
    unproven: list[dict[str, str]] = []
    blob_dst_root = backup_root / "blob"
    # A missing source-owned blob enters the package only as the exact bytes
    # its acquisition source still holds. The closed package is then complete
    # by itself: verification and restore never consult the source again.
    package_store = BlobStore(blob_dst_root)
    recovered: set[str] = set()

    def recover(blob_hash: str, size_bytes: int, source: IO[bytes]) -> bool:
        prepared = stage_exact_blob(package_store, source, blob_hash=blob_hash, size_bytes=size_bytes)
        if prepared is None:
            return False
        package_store.publish_prepared(prepared)
        recovered.add(blob_hash)
        return True

    try:
        reference_evidence["recoverability_proofs"] = _source_recoverability_proofs(
            source_db,
            root=source_blob_root.parent,
            missing_hashes=missing_hashes & source_hashes,
            unproven=unproven,
            zip_payload_cache={},
            recover=recover,
        )
    finally:
        # The package namespace has no concurrent publisher, so its staging
        # workspace is removed once empty; a leftover file fails the backup.
        if package_store.staging_root.is_dir():
            package_store.staging_root.rmdir()
    reference_evidence["recoverability_unproven"] = unproven
    _write_blob_reference_evidence(backup_root, reference_evidence)
    if not hashes:
        return 0, 0, debt_report

    count = 0
    size = 0
    copied_inventory: list[dict[str, object]] = []
    missing_reserved: list[str] = []
    for hash_hex in sorted(hashes):
        dst = blob_dst_root / hash_hex[:2] / hash_hex[2:]
        if hash_hex not in recovered:
            src = store.blob_path(hash_hex)
            if not src.exists():
                if inventory[hash_hex] == {"reserved"}:
                    missing_reserved.append(hash_hex)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        count += 1
        copied_size = dst.stat().st_size
        size += copied_size
        copied_inventory.append(
            {
                "blob_hash": hash_hex,
                "size_bytes": copied_size,
                "protection": sorted(inventory[hash_hex]),
            }
        )
    (backup_root / "blob-inventory.json").write_text(
        json.dumps(copied_inventory, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if missing_reserved:
        warnings.append(
            "source-tier publication reservations missing blob bytes: "
            f"{len(missing_reserved)} total"
            + (
                f" (sample: {', '.join(missing_reserved[:_MISSING_BLOB_WARNING_SAMPLE_LIMIT])})"
                if missing_reserved
                else ""
            )
        )
    if recovered:
        warnings.append(
            f"source-tier referenced blobs recovered into the package from their acquisition sources: {len(recovered)}"
        )
    if debt_report.missing_referenced_blobs:
        _write_blob_reference_debt_report(backup_root, debt_report)
        sample = ", ".join(debt_report.sample)
        warnings.append(
            "source-tier referenced blobs missing: "
            f"{debt_report.missing_referenced_blobs} total"
            + (f" (sample: {sample})" if sample else "")
            + "; details: blob-reference-debt.json"
            + " (this counts source.db canonical liveness -- unfetched"
            " index-tier attachments with a NULL blob_hash are never counted"
            " here; archive verification reports attachment coverage"
            " for attachment-tier acquisition state)"
        )
    return count, size, debt_report


def _write_manifest(
    *,
    backup_root: Path,
    mode: str,
    profile: BackupProfile,
    backed_up_files: list[str],
    included_tiers: list[str],
    omitted_tiers: list[str],
    blob_count: int,
    blob_size: int,
    warnings: list[str],
    archive_root_source_identity: dict[str, object],
    tier_source_fingerprints: dict[str, dict[str, object]],
    archive_authority_files: list[str],
    blob_reference_debt: BlobReferenceDebtReport | None = None,
) -> None:
    manifest = {
        "format": "polylogue-backup-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "profile": profile,
        "backed_up_files": backed_up_files,
        "included_tiers": included_tiers,
        "omitted_tiers": omitted_tiers,
        "blob_count": blob_count,
        "blob_size_bytes": blob_size,
        "blob_inventory_file": "blob-inventory.json",
        "blob_reference_evidence_file": _BLOB_REFERENCE_EVIDENCE_FILE,
        "archive_root_source_identity": archive_root_source_identity,
        "tier_source_fingerprints": tier_source_fingerprints,
        "archive_authority_files": archive_authority_files,
        "warnings": warnings,
    }
    if blob_reference_debt is not None:
        manifest["blob_reference_debt"] = blob_reference_debt.to_dict()
    (backup_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _backup_archive(
    *, output_dir: Path, started: float, profile: BackupProfile, archive_root_path: Path
) -> BackupResult:
    root = archive_root_path
    included_tiers = {
        tier: path
        for tier, path in _profile_archive_tiers(root, profile).items()
        if path.exists() or tier not in _optional_profile_tiers(profile)
    }
    omitted_tiers = {tier: path for tier, path in _all_archive_tiers(root).items() if tier not in included_tiers}
    warnings = _check_prerequisites(profile=profile, archive_root_path=root)
    if _has_backup_error(warnings):
        return BackupResult(
            ok=False,
            backup_mode="archive_file_set",
            backup_profile=profile,
            error=warnings[0],
            elapsed_s=round(time.monotonic() - started, 3),
            warnings=warnings,
            omitted_tiers=[f"{tier}.db" for tier in omitted_tiers],
        )

    ts = _timestamp()
    backup_root = output_dir / f"polylogue-archive-{ts}"
    backup_root.mkdir(parents=False, exist_ok=False)

    db_size = 0
    backed_up_files: list[str] = []
    tier_source_fingerprints: dict[str, dict[str, object]] = {}
    source_exclusion: AbstractContextManager[object] = nullcontext()
    if "source" in included_tiers:
        from polylogue.storage.blob_publication import exclude_archive_blob_publishers

        source_exclusion = exclude_archive_blob_publishers(included_tiers["source"])
    with source_exclusion:
        for tier, src in included_tiers.items():
            dst = backup_root / f"{tier}.db"
            copied_size, fingerprint = _backup_sqlite(src, dst, archive_root_path=root)
            db_size += copied_size
            tier_source_fingerprints[f"{tier}.db"] = fingerprint
            backed_up_files.append(str(dst))

        try:
            source_assertion = (
                _copy_source_declared_absent_assertion(
                    root / "source.db", backup_root, fingerprint=tier_source_fingerprints["source.db"]
                )
                if "source" in included_tiers
                else None
            )
        except RuntimeError as exc:
            # Declaration refusals remain failed package results, as when the
            # verifier authenticated a byte-identical main-file copy.
            return BackupResult(
                ok=False,
                output_path=str(backup_root),
                backup_profile=profile,
                db_size_bytes=db_size,
                elapsed_s=round(time.monotonic() - started, 3),
                error=str(exc),
                warnings=warnings,
                backed_up_files=backed_up_files,
                omitted_tiers=[f"{tier}.db" for tier in omitted_tiers],
            )

        blob_reference_debt: BlobReferenceDebtReport | None = None
        if "source" in included_tiers:
            blob_count, blob_size, blob_reference_debt = _copy_referenced_blobs(
                source_db=backup_root / "source.db",
                source_blob_root=root / "blob",
                index_db=(backup_root / "index.db" if "index" in included_tiers else None),
                backup_root=backup_root,
                warnings=warnings,
            )
        else:
            blob_count = 0
            blob_size = 0

    # Preserve original birth and numbered history proof for the included
    # durable tiers. Relocated receipts retain their original bindings:
    # copying them never admits a different inode as the live archive.
    archive_authority_files: list[str] = []
    for relative_name in _archive_authority_file_names(root, included_tiers):
        source = root / relative_name
        if not (source.exists() or source.is_symlink()):
            continue
        if relative_name == ".polylogue-format.json":
            durable_tiers = {"source", "user", "audit"}
            durable_names = {f"{tier}.db" for tier in durable_tiers}
            if not durable_tiers.issubset(included_tiers) or not all((root / name).is_file() for name in durable_names):
                # A partial/adoption backup cannot carry a completed marker:
                # restoring it without every durable leaf would make startup
                # reject the otherwise valid receipt-backed recovery path.
                continue
        _require_regular_backup_artifact(source, backup_root=root, label="archive authority")
        destination = backup_root / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        archive_authority_files.append(relative_name)
    if blob_count:
        backed_up_files.append(str(backup_root / "blob"))
    backed_up_files.extend(str(backup_root / name) for name in archive_authority_files)

    omitted = [f"{tier}.db" for tier in omitted_tiers]
    _write_manifest(
        backup_root=backup_root,
        mode="archive_file_set",
        profile=profile,
        backed_up_files=backed_up_files,
        included_tiers=[f"{tier}.db" for tier in included_tiers],
        omitted_tiers=omitted,
        blob_count=blob_count,
        blob_size=blob_size,
        warnings=warnings,
        archive_root_source_identity=_archive_root_source_identity(root),
        tier_source_fingerprints=tier_source_fingerprints,
        archive_authority_files=archive_authority_files,
        blob_reference_debt=blob_reference_debt,
    )
    backed_up_files.append(str(backup_root / "manifest.json"))
    if (backup_root / "blob-inventory.json").exists():
        backed_up_files.append(str(backup_root / "blob-inventory.json"))
    if (backup_root / _BLOB_REFERENCE_EVIDENCE_FILE).exists():
        backed_up_files.append(str(backup_root / _BLOB_REFERENCE_EVIDENCE_FILE))
    if source_assertion is not None:
        backed_up_files.append(str(source_assertion))
    if blob_reference_debt is not None and blob_reference_debt.missing_referenced_blobs:
        backed_up_files.append(str(backup_root / "blob-reference-debt.json"))

    return BackupResult(
        ok=True,
        output_path=str(backup_root),
        backup_mode="archive_file_set",
        backup_profile=profile,
        db_size_bytes=db_size,
        blob_count=blob_count,
        blob_size_bytes=blob_size,
        elapsed_s=round(time.monotonic() - started, 3),
        warnings=warnings,
        backed_up_files=backed_up_files,
        omitted_tiers=omitted,
    )


def _sqlite_integrity_ok(path: Path) -> bool:
    conn = _open_backup_readonly_connection(path, immutable=True, timeout_class="offline-bulk")
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        return row is not None and row[0] == "ok"
    finally:
        conn.close()


def _verify_backup_result(result: BackupResult) -> None:
    if result.output_path is None:
        result.verified = False
        result.verification = {"ok": False, "error": "backup has no output path"}
        result.ok = False
        return

    output_path = Path(result.output_path)
    _remove_verification_receipt(output_path)
    try:
        verification = _verify_archive_file_set_backup(output_path)
    except Exception as exc:
        verification = {"ok": False, "error": str(exc)}

    result.verification = verification
    result.verified = bool(verification.get("ok"))
    if not result.verified:
        result.ok = False
        result.error = str(verification.get("error") or "backup verification failed")
        return
    try:
        receipt_path = _write_successful_verification_receipt(output_path, verification)
    except Exception as exc:
        _remove_verification_receipt(output_path)
        result.ok = False
        result.verified = False
        result.error = f"backup verification receipt write failed: {exc}"
        result.verification = {**verification, "ok": False, "error": result.error}
        return
    result.verification = {
        **{key: value for key, value in verification.items() if key != "receipt_evidence"},
        "receipt_path": str(receipt_path),
    }
    result.backed_up_files.append(str(receipt_path))


def _backup_verification_scratch_parent(path: Path) -> Path | None:
    """Choose scratch placement near the backup to avoid root ``/tmp`` I/O."""
    from polylogue.config import load_polylogue_config

    configured_tmpdir = load_polylogue_config().backup_verify_tmpdir
    candidates = (path.parent, Path(configured_tmpdir) if configured_tmpdir else None, Path("/realm/tmp"))
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if candidate.is_dir():
            return candidate
    return None


def _copy_backup_artifact_to_scratch(source: Path, scratch_root: Path) -> Path:
    _require_real_backup_directory(source, label="backup output")
    # copytree breaks hard links. Admit the original inventory before that
    # transformation, without another full content-hashing pass.
    for artifact in source.rglob("*"):
        if stat.S_ISDIR(artifact.lstat().st_mode):
            _require_real_backup_directory(artifact, label="backup artifact directory")
        else:
            _require_regular_backup_artifact(artifact, backup_root=source, label="backup artifact")
    restore_root = scratch_root / "restore"
    shutil.copytree(source, restore_root, symlinks=True)
    return restore_root


def _remove_verification_receipt(backup_root: Path) -> None:
    receipt = backup_root / _VERIFICATION_RECEIPT_FILE
    if receipt.exists() or receipt.is_symlink():
        receipt.unlink()
        sync_directory(backup_root)


def _verify_archive_file_set_backup(path: Path) -> dict[str, object]:
    scratch_parent = _backup_verification_scratch_parent(path)
    with tempfile.TemporaryDirectory(prefix="polylogue-backup-verify-", dir=scratch_parent) as raw_tmp:
        restored = _copy_backup_artifact_to_scratch(path, Path(raw_tmp))
        if not restored.is_dir():
            return {"ok": False, "mode": "archive_file_set", "error": "backup output is not a directory"}
        copied_sidecars = _sqlite_sidecar_paths(restored)

        manifest_path = restored / "manifest.json"
        if not manifest_path.exists() and not manifest_path.is_symlink():
            return {"ok": False, "mode": "archive_file_set", "error": "manifest.json is missing"}
        _require_regular_backup_artifact(manifest_path, backup_root=restored, label="backup manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        included_tiers = [
            str(item) for item in manifest.get("included_tiers", ("source.db", "user.db", "embeddings.db"))
        ]
        omitted_tiers = [str(item) for item in manifest.get("omitted_tiers", ("index.db", "ops.db"))]
        tier_integrity: dict[str, bool] = {}
        for name in included_tiers:
            if not name.endswith(".db"):
                continue
            tier_path = restored / name
            if not tier_path.exists() and not tier_path.is_symlink():
                tier_integrity[name.removesuffix(".db")] = False
                continue
            _require_regular_backup_artifact(tier_path, backup_root=restored, label="backup tier")
            _reject_sqlite_sidecars(tier_path)
            tier_integrity[name.removesuffix(".db")] = _sqlite_integrity_ok(tier_path)
        authority_files = manifest.get("archive_authority_files", [])
        allowed_authority_files = _archive_authority_file_names(restored, (Path(name).stem for name in included_tiers))
        if not isinstance(authority_files, list) or any(
            not isinstance(item, str) or item not in allowed_authority_files for item in authority_files
        ):
            raise RuntimeError("backup manifest has invalid archive authority file declarations")
        for relative_name in authority_files:
            authority_path = restored / relative_name
            _require_regular_backup_artifact(authority_path, backup_root=restored, label="backup archive authority")
        if ".polylogue-format.json" in authority_files and all(
            (restored / f"{tier}.db").is_file() for tier in ("source", "user", "audit")
        ):
            assert_archive_format_authority(restored)
        omitted_absent = all(
            not (restored / name).exists() and not (restored / name).is_symlink() for name in omitted_tiers
        )
        blob_count = int(manifest.get("blob_count", 0) or 0)
        inventory_path = restored / str(manifest.get("blob_inventory_file", "blob-inventory.json"))
        if inventory_path.exists() or inventory_path.is_symlink():
            _require_regular_backup_artifact(
                inventory_path,
                backup_root=restored,
                label="backup blob inventory",
            )
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        else:
            inventory = []
        inventory_blobs = {
            str(item["blob_hash"]): int(item["size_bytes"])
            for item in inventory
            if isinstance(item, dict) and "blob_hash" in item and "size_bytes" in item
        }
        restored_blob_paths = _regular_backup_blob_files(restored)
        restored_blob_count = len(restored_blob_paths)
        restored_hashes: dict[str, int] = {}
        verified_blob_file_hashes: dict[str, tuple[int, str]] = {}
        hashes_valid = True
        for blob_path in restored_blob_paths:
            blob_hash = blob_path.parent.name + blob_path.name
            # Blob content comes from provider exports and attachments, which
            # the import pipeline admits at sizes far above what belongs in
            # memory at once. Size and digest are both computable
            # incrementally, so stream rather than materializing the blob.
            payload_size = 0
            digest = hashlib.sha256()
            with blob_path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    payload_size += len(chunk)
                    digest.update(chunk)
            restored_hashes[blob_hash] = payload_size
            payload_hash = digest.hexdigest()
            hashes_valid = hashes_valid and payload_hash == blob_hash
            verified_blob_file_hashes[str(blob_path.relative_to(restored))] = (payload_size, payload_hash)
        blobs_ok = (
            restored_blob_count == blob_count
            and len(inventory_blobs) == blob_count
            and restored_hashes == inventory_blobs
            and hashes_valid
        )
        restored_hash_set = set(restored_hashes)
        source_included = (restored / "source.db").exists()
        index_path = restored / "index.db"
        reference_evidence_ok = True
        source_scope_ok = True
        expected_reference_blobs: set[str] = set()
        expected_attachment_hashes: set[str] = set()
        observed_attachment_hashes: set[str] = set()
        recovered_source_hashes: set[str] = set()
        unproven_hashes: set[str] = set()
        if source_included:
            evidence_path = restored / str(manifest.get("blob_reference_evidence_file", _BLOB_REFERENCE_EVIDENCE_FILE))
            if not evidence_path.exists() and not evidence_path.is_symlink():
                raise RuntimeError("backup blob reference evidence is missing")
            _require_regular_backup_artifact(
                evidence_path, backup_root=restored, label="backup blob reference evidence"
            )
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            # Refuses malformed owner evidence; the required set comes from the package's own tiers.
            _expected_blob_hashes_from_evidence(evidence)
            source_evidence_hashes = {
                blob_hash for hashes in evidence["source_owner_hashes"].values() for blob_hash in hashes
            }
            recoverability_proofs = evidence.get("recoverability_proofs", [])
            if not isinstance(recoverability_proofs, list):
                reference_evidence_ok = False
                recoverability_proofs = []
            recoverability_unproven = evidence.get("recoverability_unproven", [])
            if not isinstance(recoverability_unproven, list):
                reference_evidence_ok = False
                recoverability_unproven = []
            for proof in recoverability_proofs:
                if not isinstance(proof, dict):
                    reference_evidence_ok = False
                    continue
                blob_hash_value = proof.get("blob_hash")
                kind = proof.get("kind")
                if (
                    not isinstance(blob_hash_value, str)
                    or blob_hash_value not in source_evidence_hashes
                    or not isinstance(proof.get("source_path"), str)
                    or kind not in _RECOVERY_PROOF_KINDS
                    or blob_hash_value in recovered_source_hashes
                ):
                    reference_evidence_ok = False
                    continue
                # A proof records bytes the backup recovered into this package
                # from their acquisition source. The package must carry them;
                # the source itself is never replayed here, because it can
                # change or vanish after the receipt is written.
                if blob_hash_value not in restored_hash_set:
                    reference_evidence_ok = False
                    continue
                recovered_source_hashes.add(blob_hash_value)
            for failure in recoverability_unproven:
                if not isinstance(failure, dict):
                    reference_evidence_ok = False
                    continue
                failure_blob_hash = failure.get("blob_hash")
                failure_kind = failure.get("kind")
                if (
                    not isinstance(failure_blob_hash, str)
                    or failure_blob_hash not in source_evidence_hashes
                    or failure_blob_hash in unproven_hashes
                    or failure_kind not in _RECOVERABILITY_FAILURE_KINDS
                ):
                    reference_evidence_ok = False
                    continue
                unproven_hashes.add(failure_blob_hash)
            expected_attachment_hashes = set(evidence["index_attachment_hashes"])
            if evidence["index_attachment_evidence"] == "consulted":
                if not index_path.exists():
                    raise RuntimeError("backup index attachment evidence was consulted but index.db is missing")
                observed_attachment_hashes = _index_attachment_hashes(index_path)
                reference_evidence_ok = (
                    reference_evidence_ok and observed_attachment_hashes == expected_attachment_hashes
                )
            closure = package_blob_closure(restored)
            reference_evidence_ok = reference_evidence_ok and closure.source_hashes == source_evidence_hashes
            if not closure.declared_absent.issubset(closure.source_hashes):
                reference_evidence_ok = False
            expected_reference_blobs = set(closure.required) | expected_attachment_hashes
            expected_unproven_hashes = source_evidence_hashes - restored_hash_set
            reference_evidence_ok = reference_evidence_ok and unproven_hashes == expected_unproven_hashes
            if closure.declared_absent_asserted:
                source_scope_ok = bool(closure.effective_source_hashes)
        missing_canonical_blobs = expected_reference_blobs - restored_hash_set
        canonical_blobs_resolved = not source_included or (
            not missing_canonical_blobs and reference_evidence_ok and source_scope_ok
        )
        ok = all(tier_integrity.values()) and omitted_absent and blobs_ok and canonical_blobs_resolved
        _discard_scratch_verification_sidecars(restored, copied=copied_sidecars)
        receipt_evidence = _receipt_evidence(restored, verified_file_hashes=verified_blob_file_hashes) if ok else None
        return {
            "ok": ok,
            "mode": "archive_file_set",
            "profile": manifest.get("profile", "rebuildable_cache_exclude"),
            "tier_integrity": tier_integrity,
            "omitted_tiers_absent": omitted_absent,
            "manifest_blob_count": blob_count,
            "restored_blob_count": restored_blob_count,
            "blob_inventory_exact": blobs_ok,
            "canonical_blobs_resolved": canonical_blobs_resolved,
            "missing_canonical_blob_count": len(missing_canonical_blobs),
            "recovered_source_blob_count": len(recovered_source_hashes),
            "unproven_source_blob_count": len(unproven_hashes),
            "reference_evidence_resolved": reference_evidence_ok,
            "source_effective_scope_nonempty": source_scope_ok,
            "expected_index_attachment_count": len(expected_attachment_hashes),
            "observed_index_attachment_count": len(observed_attachment_hashes),
            "scratch_restore": "temporary",
            "scratch_parent": str(Path(raw_tmp).parent),
            "receipt_evidence": receipt_evidence,
        }


def _receipt_tier_artifacts(
    backup_root: Path,
    manifest: dict[str, object],
    *,
    file_evidence: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    fingerprints = manifest.get("tier_source_fingerprints")
    source_fingerprints = fingerprints if isinstance(fingerprints, dict) else {}
    artifacts: list[dict[str, object]] = []
    for name in _json_str_list(manifest.get("included_tiers")):
        filename = str(name)
        if not filename.endswith(".db"):
            continue
        path = backup_root / filename
        if not path.exists() and not path.is_symlink():
            continue
        _require_regular_backup_artifact(path, backup_root=backup_root, label="backup tier")
        _reject_sqlite_sidecars(path)
        source_fingerprint = source_fingerprints.get(filename)
        evidence = file_evidence.get(filename, {})
        artifact = {
            "tier": filename.removesuffix(".db"),
            "path": filename,
            "size_bytes": evidence.get("size_bytes"),
            "sha256": evidence.get("sha256"),
            "user_version": _sqlite_user_version(path),
            "source_fingerprint": source_fingerprint,
        }
        snapshot = source_fingerprint.get("snapshot") if isinstance(source_fingerprint, dict) else None
        if not isinstance(snapshot, dict) or any(
            artifact[field] != snapshot.get(field) for field in ("size_bytes", "sha256", "user_version")
        ):
            raise RuntimeError(f"{filename} backup artifact does not match its pinned snapshot fingerprint")
        source_path_value = source_fingerprint.get("path")
        if isinstance(source_path_value, str) and source_path_value:
            source_path = Path(source_path_value)
            if source_path.exists() and path.samefile(source_path):
                raise RuntimeError(f"{filename} backup artifact aliases its live source tier")
        artifacts.append(artifact)
    return artifacts


def _receipt_blob_inventory(
    backup_root: Path,
    manifest: dict[str, object],
    *,
    file_evidence: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], str]:
    inventory_file = str(manifest.get("blob_inventory_file", "blob-inventory.json"))
    inventory_path = backup_root / inventory_file
    declared: dict[str, dict[str, object]] = {}
    if inventory_path.exists() or inventory_path.is_symlink():
        _require_regular_backup_artifact(
            inventory_path,
            backup_root=backup_root,
            label="backup blob inventory",
        )
        raw_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        if isinstance(raw_inventory, list):
            for item in raw_inventory:
                if isinstance(item, dict) and "blob_hash" in item:
                    declared[str(item["blob_hash"])] = item

    rows: list[dict[str, object]] = []
    for blob_path in _regular_backup_blob_files(backup_root):
        blob_hash = blob_path.parent.name + blob_path.name
        declared_item = declared.get(blob_hash, {})
        protection = declared_item.get("protection")
        relative_path = str(blob_path.relative_to(backup_root))
        evidence = file_evidence.get(relative_path, {})
        rows.append(
            {
                "blob_hash": blob_hash,
                "path": relative_path,
                "size_bytes": evidence.get("size_bytes"),
                "sha256": evidence.get("sha256"),
                "protection": _json_str_list(protection),
            }
        )
    rows.sort(key=lambda item: str(item["blob_hash"]))
    return rows, _canonical_json_sha256(rows)


def _inventory_file_evidence(
    backup_root: Path,
    manifest: dict[str, object],
    *,
    file_evidence: dict[str, dict[str, object]],
) -> dict[str, object]:
    filename = str(manifest.get("blob_inventory_file", "blob-inventory.json"))
    path = backup_root / filename
    if not path.exists() and not path.is_symlink():
        return {"path": filename, "present": False, "size_bytes": 0, "sha256": None}
    _require_regular_backup_artifact(path, backup_root=backup_root, label="backup blob inventory")
    evidence = file_evidence.get(filename, {})
    return {
        "path": filename,
        "present": True,
        "size_bytes": evidence.get("size_bytes"),
        "sha256": evidence.get("sha256"),
    }


def _receipt_evidence(
    backup_root: Path,
    *,
    verified_file_hashes: Mapping[str, tuple[int, str]] | None = None,
) -> dict[str, object]:
    _require_real_backup_directory(backup_root, label="backup root")
    artifact_inventory = _backup_artifact_inventory(backup_root, verified_file_hashes=verified_file_hashes)
    file_evidence = {str(item["path"]): item for item in artifact_inventory if item.get("type") == "file"}
    manifest_path = backup_root / "manifest.json"
    _require_regular_backup_artifact(manifest_path, backup_root=backup_root, label="backup manifest")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    blobs, blob_inventory_root_sha256 = _receipt_blob_inventory(
        backup_root,
        manifest,
        file_evidence=file_evidence,
    )
    return {
        "manifest_size_bytes": len(manifest_bytes),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "included_tiers": _json_str_list(manifest.get("included_tiers")),
        "artifact_inventory": artifact_inventory,
        "tier_artifacts": _receipt_tier_artifacts(backup_root, manifest, file_evidence=file_evidence),
        "blob_inventory_file": _inventory_file_evidence(
            backup_root,
            manifest,
            file_evidence=file_evidence,
        ),
        "blob_inventory_root_sha256": blob_inventory_root_sha256,
        "blobs": blobs,
    }


def _write_successful_verification_receipt(backup_root: Path, verification: dict[str, object]) -> Path:
    manifest_path = backup_root / "manifest.json"
    verified_evidence = verification.get("receipt_evidence")
    if not isinstance(verified_evidence, dict):
        raise RuntimeError("scratch verification did not produce receipt evidence")
    try:
        current_evidence = _receipt_evidence(backup_root)
    except RuntimeError as exc:
        raise RuntimeError(f"backup changed after scratch verification: {exc}") from exc
    if current_evidence != verified_evidence:
        raise RuntimeError("backup changed after scratch verification")
    # Reading the accepted bytes is not a destination persistence barrier.
    # Complete their physical closure before the receipt can authorize them.
    sync_tree(backup_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_body: dict[str, object] = {
        "format": VERIFICATION_RECEIPT_FORMAT,
        "verdict": "success",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "mode": "archive_file_set",
        "profile": manifest.get("profile", "rebuildable_cache_exclude"),
        "manifest_path": "manifest.json",
        **verified_evidence,
        "verification": {
            key: value
            for key, value in verification.items()
            if key not in {"scratch_parent", "scratch_restore", "receipt_evidence"}
        },
    }
    authority_paths: dict[str, Path] = {}
    artifacts = verified_evidence.get("tier_artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict) or artifact.get("tier") not in {"source", "user", "audit"}:
                continue
            fingerprint = artifact.get("source_fingerprint")
            source_path = fingerprint.get("path") if isinstance(fingerprint, dict) else None
            if isinstance(source_path, str) and source_path:
                authority_paths[str(artifact["tier"])] = Path(source_path).resolve(strict=False)
    receipt = sign_verification_receipt(receipt_body, authority_paths=authority_paths)
    receipt_path = backup_root / _VERIFICATION_RECEIPT_FILE
    atomic_replace(receipt_path, json.dumps(receipt, indent=2, sort_keys=True).encode())
    return receipt_path


def create_backup_package(
    *,
    output_dir: Path,
    archive_root_path: Path,
    profile: BackupProfile,
    archive_owner: OwnedArchiveLocation,
    verify: bool = True,
    started: float | None = None,
) -> BackupResult:
    """Create the authenticated package under an existing archive writer hold."""
    assert_owns_archive_location(archive_owner, ArchiveLocation.resolve(archive_root_path))
    require_write_lease("archive backup package", archive_root=archive_root_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = _backup_archive(
        output_dir=output_dir,
        started=time.monotonic() if started is None else started,
        profile=profile,
        archive_root_path=archive_root_path,
    )
    if verify and result.ok and result.output_path is not None:
        _verify_backup_result(result)
    elif result.ok and result.output_path is not None:
        sync_tree(Path(result.output_path))
    return result


def create_pre_migration_backup(
    archive_root: Path,
    *,
    tier: str,
    current_version: int,
    target_version: int,
    archive_owner: OwnedArchiveLocation,
) -> Path:
    """Authenticate the exact tier bytes the next numbered train will change."""
    output_dir = (
        archive_root
        / ".maintenance-state"
        / "pre-migration-backups"
        / f"{tier}-v{current_version}-to-v{target_version}-{int(time.time() * 1000)}"
    )
    result = create_backup_package(
        output_dir=output_dir,
        verify=True,
        profile="rebuildable_cache_exclude" if tier == "source" else "user_overlays",
        archive_root_path=archive_root,
        archive_owner=archive_owner,
    )
    if not result.ok or result.output_path is None:
        raise RuntimeError(f"pre-migration backup of {tier}.db failed; refusing to migrate: {result.error}")
    return Path(result.output_path) / "manifest.json"

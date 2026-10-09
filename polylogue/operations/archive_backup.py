"""Backup and restore operations adapting the authenticated storage owner."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, cast

from polylogue.core.write_lease import require_write_lease, write_lease
from polylogue.paths import archive_root
from polylogue.storage import backup_package as package
from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation
from polylogue.storage.backup_attestation import VERIFICATION_RECEIPT_FORMAT
from polylogue.storage.backup_blob_closure import SOURCE_DECLARED_ABSENT_FILE, package_blob_closure
from polylogue.storage.backup_package import BACKUP_PROFILES, BackupProfile, BackupResult

if TYPE_CHECKING:
    from polylogue.operations.daemon_protocol import DaemonOperationEnvelope, DaemonOperationRequest
    from polylogue.operations.operation_context_types import OperationContext


def _require_exclusive_archive_ownership(root: Path) -> None:
    """Refuse a snapshot of tiers a resident ``polylogued`` is writing.

    A backup is a writer, not a reader: :func:`package._backup_sqlite` opens each live
    tier through ``open_isolated_write_connection``, drains its WAL with a
    ``TRUNCATE`` checkpoint and holds ``BEGIN IMMEDIATE`` across the copy, and
    :func:`package._checkpoint_sqlite_for_snapshot` states the precondition outright --
    "an exclusive boundary that already requires no concurrent writer".

    Nothing in this module established that. :func:`backup_archive` mints its
    own ``write_lease("maintenance.backup")``, which satisfies every
    ``require_write_lease`` in this process and lets the armed connection guard
    pass the write through, so the in-process lease proves nothing about a
    *second process*. Beside a live daemon the snapshot truncated the daemon's
    WAL underneath it and retried ``_SNAPSHOT_LOCK_ATTEMPTS`` times for the
    lock.

    The check lives here, at the function that mints the lease, rather than in
    ``polylogue/cli/commands/backup.py`` where it first landed. That placement
    covered exactly one caller: ``backup_archive`` is public API
    (``__all__``), and an embedded Python process importing it reached the
    whole truncating snapshot with no ownership check at all -- the standalone
    Python entry point AC1's coverage receipt names (polylogue-8qm4k AC1,
    polylogue-5vps8 AC1, polylogue-re6s3 AC1).

    ``check_only`` never reaches here: it opens nothing writable, and a
    prerequisite check is what an operator runs *before* stopping the daemon.
    """
    from polylogue.core.write_lease import coordinator_write_lease_active
    from polylogue.maintenance.offline_guard import (
        ArchiveWriterOwnershipError,
        ArchiveWriterOwnershipUndecidableError,
        DaemonResidencyUndecidableError,
        resident_daemon_pid,
    )

    if coordinator_write_lease_active():
        # The caller's daemon lease must own this exact archive, not merely
        # some archive in the current process.
        require_write_lease("maintenance.backup", archive_root=root)
        return
    try:
        pid = resident_daemon_pid(root)
    except DaemonResidencyUndecidableError as exc:
        raise ArchiveWriterOwnershipUndecidableError(
            f"cannot prove whether a resident daemon owns {root}: {exc}. Refusing to "
            "checkpoint and write-lock live tiers beside a writer this platform cannot see",
            archive_root=root,
        ) from exc
    if pid is None:
        return
    reason = f"polylogued PID {pid} is running for this archive"
    raise ArchiveWriterOwnershipError(
        f"refusing to back up {root}: {reason}. A backup snapshot checkpoints and "
        "write-locks each live tier, so it must own the archive exclusively. Submit the "
        "declared maintenance.backup operation to that daemon, or run "
        "`polylogue ops backup --check` to verify prerequisites without touching the tiers",
        archive_root=root,
        resident_writer=reason,
    )


def backup_archive(
    *,
    output_dir: Path,
    check_only: bool = False,
    verify: bool = False,
    profile: BackupProfile = "rebuildable_cache_exclude",
    archive_root_path: Path | None = None,
    archive_owner: OwnedArchiveLocation | None = None,
) -> BackupResult:
    """Backup the Polylogue archive.

    Archives are backed up by named durability profiles. The default
    ``rebuildable_cache_exclude`` profile preserves the historical behavior:
    source.db, user.db, embeddings.db, plus blobs referenced by source.db;
    index.db and ops.db are omitted because they are rebuildable/disposable.

    Args:
        output_dir: Target directory for the backup.
        check_only: If True, only verify prerequisites without creating a backup.
        verify: Restore the finished backup into a scratch directory and run
            integrity/smoke checks before returning.
        profile: Named backup profile controlling which archive tiers are copied.
        archive_owner: Exclusive archive ownership the caller already holds --
            the daemon at open, before it serves anything. The backup copies
            under that ownership instead of acquiring its own.
    """
    started = time.monotonic()

    root = archive_root_path or archive_root()
    if check_only:
        warnings = package._check_prerequisites(profile=profile, archive_root_path=root)
        return BackupResult(
            ok=len(warnings) == 0,
            check_only=True,
            backup_mode="archive_file_set",
            backup_profile=profile,
            warnings=warnings,
            error=warnings[0] if warnings else None,
            elapsed_s=round(time.monotonic() - started, 3),
        )

    from polylogue.core.write_lease import coordinator_write_lease_active
    from polylogue.maintenance.offline_guard import scoped_offline_archive_writer

    # The daemon's coordinator already owns a durable writer hold. A direct
    # Python caller pins both the daemon pidfile and durable anchor until its
    # checkpointing copies finish, so a later daemon cannot race this check.
    if archive_owner is not None:
        from polylogue.operations.durable_change_train import assert_holds_archive_ownership

        assert_holds_archive_ownership(archive_owner, root)
    owner_scope = (
        nullcontext()
        if coordinator_write_lease_active() or archive_owner is not None
        else scoped_offline_archive_writer(root, owner_id="maintenance.backup")
    )
    with owner_scope:
        if archive_owner is None:
            _require_exclusive_archive_ownership(root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with write_lease("maintenance.backup", archive_root=root):
            if archive_owner is not None:
                result = package.create_backup_package(
                    output_dir=output_dir,
                    started=started,
                    profile=profile,
                    archive_root_path=root,
                    archive_owner=archive_owner,
                    verify=False,
                )
            else:
                with OwnedArchiveLocation.acquire(
                    ArchiveLocation.resolve(root), owner_id="maintenance.backup", allow_reentrant=True
                ) as owned:
                    result = package.create_backup_package(
                        output_dir=output_dir,
                        started=started,
                        profile=profile,
                        archive_root_path=root,
                        archive_owner=owned,
                        verify=False,
                    )
    if verify and result.ok and result.output_path is not None:
        package._verify_backup_result(result)
    return result


def _retryable_archive_io_failure(exc: BaseException) -> bool:
    """Classify the backup family's existing named infrastructure failures."""
    cause: BaseException | None = exc
    visited: set[int] = set()
    retryable = False
    while cause is not None and id(cause) not in visited:
        visited.add(id(cause))
        if isinstance(cause, OSError):
            retryable = True
            break
        if isinstance(cause, sqlite3.OperationalError) and (getattr(cause, "sqlite_errorcode", 0) & 0xFF) in {
            sqlite3.SQLITE_BUSY,
            sqlite3.SQLITE_LOCKED,
            sqlite3.SQLITE_READONLY,
            sqlite3.SQLITE_IOERR,
            sqlite3.SQLITE_FULL,
            sqlite3.SQLITE_CANTOPEN,
            sqlite3.SQLITE_PROTOCOL,
            sqlite3.SQLITE_PERM,
        }:
            retryable = True
            break
        cause = cause.__cause__
    return retryable


async def execute_backup_operation(
    request: DaemonOperationRequest, context: OperationContext
) -> DaemonOperationEnvelope:
    """Run a backup without pinning an index reader across WAL checkpoints."""
    from polylogue.operations.daemon_execution import (
        _validate_identity,
        operation_envelope,
        validate_execution_request,
    )
    from polylogue.operations.operation_context import observe_control_authority

    request = validate_execution_request(request, context)
    runtime = context.runtime
    assert runtime is not None
    snapshot = await runtime.compute_phase(lambda: observe_control_authority(context.archive_root))
    _validate_identity(request, context, snapshot)
    payload = request.payload

    def run() -> BackupResult:
        return backup_archive(
            output_dir=Path(str(payload["output_dir"])),
            check_only=bool(payload.get("check_only", False)),
            verify=bool(payload.get("verify", False)),
            profile=cast(BackupProfile, payload.get("profile", "rebuildable_cache_exclude")),
            archive_root_path=context.archive_root,
        )

    # The first readability observation owns no backup artifact or write attempt.
    # The snapshot route rechecks under its writer authority before copying.
    try:
        preflight = await runtime.compute_phase(
            lambda: backup_archive(
                output_dir=Path(str(payload["output_dir"])),
                check_only=True,
                profile=cast(BackupProfile, payload.get("profile", "rebuildable_cache_exclude")),
                archive_root_path=context.archive_root,
            )
        )
    except (sqlite3.Error, OSError) as exc:
        retryable = _retryable_archive_io_failure(exc)
        return operation_envelope(
            request,
            context,
            snapshot=snapshot,
            outcome="failed" if retryable else "rejected",
            error={
                "code": "backup_io_fault" if retryable else type(exc).__name__,
                "detail": str(exc),
                "retryable": retryable,
            },
        )
    if payload.get("check_only"):
        result = preflight
    elif package._has_backup_error(preflight.warnings):
        result = preflight.model_copy(update={"check_only": False})
    else:
        runtime.begin_unbound_write(request, snapshot=snapshot)
        result = await runtime.write_phase("backup", run)
    detail = result.model_dump(mode="json")
    return operation_envelope(
        request,
        context,
        snapshot=snapshot,
        outcome="completed" if result.ok else "rejected",
        error=None
        if result.ok
        else {
            "code": "backup_failed",
            "detail": result.error or "backup failed",
            "retryable": False,
            "data": {"backup_result": detail},
        },
        result=(
            {
                "operation": request.operation,
                "outcome": "completed",
                "sequence": 1,
                "effect": "no-effect" if result.check_only else "committed",
                "result": detail,
            }
            if result.ok
            else None
        ),
    )


class ArchiveRestoreRefusalError(ValueError):
    """A backup cannot authorize the requested fresh operational destination."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def restore_verified_backup(*, backup_dir: Path, destination: Path) -> dict[str, object]:
    """Restore a closed authenticated package into new destination-owned inodes.

    Original receipts remain immutable detached provenance. Ordinary startup
    never interprets them as authority for a transplanted SQLite file.
    """
    from polylogue.storage.backup_attestation import verify_verification_receipt
    from polylogue.storage.sqlite.archive_population import populate_authenticated_archive
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.migration_runner import _validate_closed_backup_package

    backup_dir = backup_dir.resolve(strict=True)
    destination = destination.absolute()
    package._require_real_backup_directory(backup_dir, label="backup root")
    if destination.exists() or destination.is_symlink():
        raise ArchiveRestoreRefusalError("restore_destination_exists")
    manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((backup_dir / package._VERIFICATION_RECEIPT_FILE).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(receipt, dict):
        raise ArchiveRestoreRefusalError("restore_invalid_package")
    if manifest.get("mode") != "archive_file_set" or receipt.get("format") != VERIFICATION_RECEIPT_FORMAT:
        raise ArchiveRestoreRefusalError("restore_unsupported_package")
    included_values = manifest.get("included_tiers")
    if not isinstance(included_values, list) or not all(isinstance(value, str) for value in included_values):
        raise ArchiveRestoreRefusalError("restore_invalid_package")
    included = set(included_values)
    if not included.issubset({f"{tier.value}.db" for tier in ArchiveTier}):
        raise ArchiveRestoreRefusalError("restore_unsupported_package")
    debt = manifest.get("blob_reference_debt")
    original_missing_blobs = debt.get("missing_referenced_blobs", 0) if isinstance(debt, dict) else 0
    if type(original_missing_blobs) is not int or original_missing_blobs < 0:
        raise ArchiveRestoreRefusalError("restore_invalid_package")
    if not {"source.db", "user.db", "audit.db"}.issubset(included):
        # Overlay and diagnostics packages retain their evidence contract;
        # they cannot supply a complete source/audit continuity authority.
        raise ArchiveRestoreRefusalError("restore_partial_durable_core")

    def validate() -> dict[str, dict[str, object]]:
        artifacts = _validate_closed_backup_package(
            backup_dir, manifest, receipt, target_tier=None, live_tier_path=None
        )
        for tier in ("source", "user", "audit"):
            fingerprint = artifacts[tier]["source_fingerprint"]
            if not isinstance(fingerprint, dict) or not isinstance(fingerprint.get("path"), str):
                raise ArchiveRestoreRefusalError("restore_missing_original_authority")
            verify_verification_receipt(receipt, tier=tier, live_tier_path=Path(fingerprint["path"]))
        return artifacts

    def validate_source_files() -> None:
        validate()

    artifacts = validate()
    # The manifest's debt describes the original live store before backup
    # recovery. Only the authenticated package determines unrestored bytes.
    closure = package_blob_closure(backup_dir)
    carried_blobs = {str(blob["blob_hash"]) for blob in receipt["blobs"]}
    missing_blobs = len((closure.source_hashes | closure.index_hashes | closure.reservations) - carried_blobs)
    original_identities: dict[ArchiveTier, str] = {}
    for tier in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT):
        fingerprint = artifacts[tier.value]["source_fingerprint"]
        if (
            not isinstance(fingerprint, dict)
            or type(fingerprint.get("device")) is not int
            or type(fingerprint.get("inode")) is not int
        ):
            raise ArchiveRestoreRefusalError("restore_missing_original_identity")
        original_identities[tier] = hashlib.sha256(
            f"dev:{fingerprint['device']}:ino:{fingerprint['inode']}".encode()
        ).hexdigest()
    source_files = tuple(
        (str(item["path"]), int(item["size_bytes"]), str(item["sha256"]))
        for item in receipt["artifact_inventory"]
        if item.get("type") == "file"
    )
    source_manifest_id = str(receipt["manifest_sha256"])
    from polylogue.storage.sqlite.population_admission import (
        POPULATION_PENDING,
        ArchivePopulationDestinationExistsError,
        reserve_population_destination,
    )

    try:
        with reserve_population_destination(destination, source_manifest_id=source_manifest_id) as admission:
            destination = admission.root
            entries = {path.name for path in destination.iterdir()}
            from polylogue.storage.sqlite.write_lease import ARCHIVE_WRITE_CUSTODY_LOCK_NAME

            if entries != {
                POPULATION_PENDING,
                "daemon.pid",
                ".archive-ownership.lock",
                ARCHIVE_WRITE_CUSTODY_LOCK_NAME,
            }:
                raise ArchiveRestoreRefusalError("restore_destination_reservation_conflict")
            shutil.copytree(backup_dir, destination, dirs_exist_ok=True)
            proof = populate_authenticated_archive(
                backup_dir,
                destination,
                source_manifest_id=source_manifest_id,
                source_files=source_files,
                retained_artifact_reference=False,
                validate_source_files=validate_source_files,
                original_tier_identities=original_identities,
            )
            if proof is None:
                raise ArchiveRestoreRefusalError("restore_missing_format_authority")
            # Compare every unchanged package file, including all copied
            # blob bytes, before retiring the pending admission fence.
            for relative, size, digest in source_files:
                if relative in proof.replaced_paths:
                    continue
                path = destination / relative
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or path.stat().st_size != size
                    or package._sha256_file(path) != digest
                ):
                    raise ArchiveRestoreRefusalError("restore_destination_file_mismatch")
            provenance = (
                destination / ".archive-population-provenance" / hashlib.sha256(source_manifest_id.encode()).hexdigest()
            )
            original_package = provenance / "original-backup"
            original_package.mkdir()
            for name in (
                "manifest.json",
                package._VERIFICATION_RECEIPT_FILE,
                "blob-inventory.json",
                package._BLOB_REFERENCE_EVIDENCE_FILE,
                "blob-reference-debt.json",
                SOURCE_DECLARED_ABSENT_FILE,
            ):
                path = destination / name
                if path.is_file():
                    path.replace(original_package / name)
            validate()
            # The deep owner already evaluated the normal final startup
            # predicate after exact row/schema population. This point
            # additionally proves the final file/blob set and source
            # package binding while the same owner still excludes writers.
    except ArchivePopulationDestinationExistsError as exc:
        raise ArchiveRestoreRefusalError("restore_destination_exists") from exc
    return {
        "destination": str(destination),
        "source_manifest_id": source_manifest_id,
        "restored_tiers": sorted(included - proof.new_derived_tiers),
        "new_empty_derived_tiers": sorted(proof.new_derived_tiers),
        "requires_convergence": sorted(proof.new_derived_tiers),
        # The canonical constructor creates all six tiers. An empty purchased
        # tier admits operations but does not recover omitted vectors.
        "unrestored_purchased_tiers": sorted({"embeddings.db"} - included),
        "unrestored_referenced_blobs": missing_blobs,
        "operational_admission": (
            "degraded" if "embeddings.db" not in included or missing_blobs or proof.new_derived_tiers else "ready"
        ),
        "original_history": ".archive-population-provenance",
    }


async def execute_restore_verified_backup_operation(
    request: DaemonOperationRequest, context: OperationContext
) -> DaemonOperationEnvelope:
    """Accept an explicit restore and populate its separately owned fresh root."""
    from polylogue.operations.daemon_execution import _validate_identity, operation_envelope, validate_execution_request
    from polylogue.operations.operation_context import observe_control_authority
    from polylogue.storage.backup_attestation import BackupAttestationError
    from polylogue.storage.sqlite.archive_population import ArchivePopulationError
    from polylogue.storage.sqlite.migration_runner import MigrationError
    from polylogue.storage.sqlite.population_admission import POPULATION_PENDING, ArchivePopulationPendingError

    request = validate_execution_request(request, context)
    runtime = context.runtime
    assert runtime is not None
    snapshot = await runtime.compute_phase(lambda: observe_control_authority(context.archive_root))
    _validate_identity(request, context, snapshot)
    runtime.begin_unbound_write(request, snapshot=snapshot)
    try:
        detail = await runtime.compute_phase(
            lambda: restore_verified_backup(
                backup_dir=Path(str(request.payload["backup_dir"])),
                destination=Path(str(request.payload["destination"])),
            )
        )
    except (
        ArchiveRestoreRefusalError,
        ArchivePopulationError,
        ArchivePopulationPendingError,
        BackupAttestationError,
        MigrationError,
        OSError,
        sqlite3.OperationalError,
        json.JSONDecodeError,
    ) as exc:
        destination = Path(str(request.payload["destination"]))
        retryable = _retryable_archive_io_failure(exc)
        error: dict[str, object] = {
            "code": "restore_io_fault" if retryable else getattr(exc, "code", "restore_invalid_evidence"),
            "retryable": retryable,
        }
        if (destination / POPULATION_PENDING).is_file():
            error["retained_pending_destination"] = str(destination)
        return operation_envelope(
            request, context, snapshot=snapshot, outcome="failed" if retryable else "rejected", error=error
        )
    return operation_envelope(
        request,
        context,
        snapshot=snapshot,
        outcome="completed",
        result={
            "operation": request.operation,
            "outcome": "completed",
            "sequence": 1,
            "effect": "committed",
            "result": detail,
        },
    )


def format_backup_result(result: BackupResult) -> list[str]:
    """Render backup result as plain-text lines."""
    lines: list[str] = []
    if result.check_only:
        if result.ok:
            lines.append("Backup prerequisites: OK")
        else:
            lines.append(f"Backup prerequisites: FAILED — {result.error}")
        for w in result.warnings:
            lines.append(f"  Warning: {w}")
        return lines

    if result.ok:
        lines.append(f"Backup complete: {result.output_path}")
        lines.append("  Mode: archive")
        lines.append(f"  Profile: {result.backup_profile}")
        if result.omitted_tiers:
            if result.backup_profile == "rebuildable_cache_exclude":
                lines.append(f"  Omitted: {', '.join(result.omitted_tiers)} (rebuildable/disposable)")
            else:
                lines.append(f"  Omitted by profile: {', '.join(result.omitted_tiers)}")
    else:
        lines.append(f"Backup failed: {result.error}")
        lines.append(f"  Partial output: {result.output_path}")

    lines.append(f"  DB size: {result.db_size_bytes / (1024**2):.1f} MB")
    if result.blob_count:
        lines.append(f"  Blobs: {result.blob_count} ({result.blob_size_bytes / (1024**2):.1f} MB)")
    if result.verified:
        lines.append("  Verification: OK")
    elif result.verification:
        lines.append(f"  Verification: FAILED — {result.verification.get('error', 'see details')}")
    lines.append(f"  Elapsed: {result.elapsed_s:.1f}s")
    for w in result.warnings:
        lines.append(f"  Warning: {w}")
    return lines


__all__ = [
    "BackupResult",
    "BACKUP_PROFILES",
    "BackupProfile",
    "backup_archive",
    "restore_verified_backup",
    "format_backup_result",
]

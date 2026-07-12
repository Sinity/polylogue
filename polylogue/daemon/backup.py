"""Backup and portability operations for the Polylogue archive.

Provides a local-first backup command for tiered archives.
Backups copy only the precious tiers plus referenced blobs: source.db, user.db,
embeddings.db, and blob files. Rebuildable index.db and disposable ops.db are
omitted by design.

Each SQLite tier is checkpointed and copied while its write lock is held, so
the copied bytes and recorded source fingerprint describe the same state.
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
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from polylogue.logging import get_logger
from polylogue.paths import archive_root
from polylogue.storage.backup_attestation import VERIFICATION_RECEIPT_FORMAT, sign_verification_receipt
from polylogue.storage.blob_integrity import BlobReferenceDebtReport, referenced_blob_hashes, scan_blob_reference_debt
from polylogue.storage.blob_store import BlobStore

logger = get_logger(__name__)

BackupProfile = Literal["full_evidence", "user_overlays", "rebuildable_cache_exclude", "diagnostics_bundle"]
BACKUP_PROFILES: tuple[BackupProfile, ...] = (
    "full_evidence",
    "user_overlays",
    "rebuildable_cache_exclude",
    "diagnostics_bundle",
)
_MISSING_BLOB_WARNING_SAMPLE_LIMIT = 10
_VERIFICATION_RECEIPT_FILE = "verification-receipt.json"
_SNAPSHOT_LOCK_ATTEMPTS = 5
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


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


def _backup_artifact_inventory(backup_root: Path) -> list[dict[str, object]]:
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
        rows.append(
            {
                "path": str(relative),
                "type": "file",
                "size_bytes": metadata.st_size,
                "sha256": _sha256_file(candidate),
            }
        )
    return rows


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sqlite_user_version(path: Path) -> int:
    with sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)


def _sqlite_source_fingerprint(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "user_version": _sqlite_user_version(path),
    }


def _json_str_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _precious_archive_tiers(root: Path) -> dict[str, Path]:
    return {
        "source": root / "source.db",
        "user": root / "user.db",
        "embeddings": root / "embeddings.db",
    }


def _omitted_archive_tiers(root: Path) -> dict[str, Path]:
    return {
        "index": root / "index.db",
        "ops": root / "ops.db",
    }


def _all_archive_tiers(root: Path) -> dict[str, Path]:
    return {
        "source": root / "source.db",
        "index": root / "index.db",
        "embeddings": root / "embeddings.db",
        "user": root / "user.db",
        "ops": root / "ops.db",
    }


def _profile_archive_tiers(root: Path, profile: BackupProfile) -> dict[str, Path]:
    all_tiers = _all_archive_tiers(root)
    if profile == "full_evidence":
        return all_tiers
    if profile == "user_overlays":
        return {"user": all_tiers["user"]}
    if profile == "diagnostics_bundle":
        return {"ops": all_tiers["ops"]}
    return {
        "source": all_tiers["source"],
        "user": all_tiers["user"],
        "embeddings": all_tiers["embeddings"],
    }


def _optional_profile_tiers(profile: BackupProfile) -> set[str]:
    if profile == "full_evidence":
        return {"ops"}
    if profile == "rebuildable_cache_exclude":
        return {"embeddings"}
    return set()


def _profile_omitted_tiers(root: Path, profile: BackupProfile) -> dict[str, Path]:
    included = set(_profile_archive_tiers(root, profile))
    return {tier: path for tier, path in _all_archive_tiers(root).items() if tier not in included}


def _archive_layout_present(root: Path) -> bool:
    return any(path.exists() for path in _all_archive_tiers(root).values())


def _readable_sqlite(path: Path) -> str | None:
    try:
        conn = sqlite3.connect(str(path))
        try:
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return str(exc)
    return None


def _check_prerequisites(*, profile: BackupProfile = "rebuildable_cache_exclude") -> list[str]:
    """Return a list of warning/error strings for backup prerequisites."""
    warnings: list[str] = []

    root = archive_root()
    if not _archive_layout_present(root):
        return [f"archive tiers not found under {root}"]

    optional_tiers = _optional_profile_tiers(profile)
    for tier, path in _profile_archive_tiers(root, profile).items():
        if not path.exists():
            if tier in optional_tiers:
                continue
            warnings.append(f"{tier}.db not found at {path}")
            continue
        error = _readable_sqlite(path)
        if error is not None:
            warnings.append(f"{tier}.db not readable: {error}")

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
    return any("not found" in warning or "not readable" in warning for warning in warnings)


def _checkpoint_sqlite_for_snapshot(conn: sqlite3.Connection, path: Path) -> None:
    row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if row is None:
        raise RuntimeError(f"could not checkpoint {path} before backup")
    busy, log_frames, checkpointed_frames = (int(value) for value in row)
    if busy or log_frames != checkpointed_frames:
        raise RuntimeError(f"could not quiesce {path} before backup")


def _backup_sqlite(src: Path, dst: Path) -> tuple[int, dict[str, object]]:
    """Copy a checkpointed tier while excluding concurrent SQLite writers."""
    live_path = src.resolve(strict=True)
    conn = sqlite3.connect(str(live_path), timeout=30.0)
    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        for _attempt in range(_SNAPSHOT_LOCK_ATTEMPTS):
            _checkpoint_sqlite_for_snapshot(conn, live_path)
            conn.execute("BEGIN IMMEDIATE")
            wal_path = live_path.with_name(f"{live_path.name}-wal")
            if wal_path.exists() and wal_path.stat().st_size:
                conn.rollback()
                continue
            fingerprint = _sqlite_source_fingerprint(live_path)
            try:
                shutil.copy2(live_path, dst)
            except Exception:
                dst.unlink(missing_ok=True)
                raise
            return dst.stat().st_size, fingerprint
        raise RuntimeError(f"could not obtain a checkpointed write-locked snapshot of {live_path}")
    finally:
        if conn.in_transaction:
            conn.rollback()
        conn.close()


def _source_blob_inventory(source_db: Path) -> dict[str, set[str]]:
    inventory = {blob_hash: {"referenced"} for blob_hash in referenced_blob_hashes(source_db, immutable=True)}
    with sqlite3.connect(f"file:{source_db}?mode=ro&immutable=1", uri=True) as conn:
        has_reservations = conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'blob_publication_reservations'"
        ).fetchone()
        if has_reservations is not None:
            for (blob_hash,) in conn.execute("SELECT DISTINCT blob_hash FROM blob_publication_reservations"):
                inventory.setdefault(bytes(blob_hash).hex(), set()).add("reserved")
    return inventory


def _source_blob_hashes(source_db: Path) -> set[str]:
    return set(_source_blob_inventory(source_db))


def _index_attachment_blob_hashes(index_db: Path) -> set[str]:
    if not index_db.exists():
        return set()
    with sqlite3.connect(f"file:{index_db}?mode=ro&immutable=1", uri=True) as conn:
        has_attachments = conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'attachments'"
        ).fetchone()
        if has_attachments is None:
            return set()
        return {
            bytes(blob_hash).hex()
            for (blob_hash,) in conn.execute("SELECT DISTINCT blob_hash FROM attachments WHERE blob_hash IS NOT NULL")
        }


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
    inventory = _source_blob_inventory(source_db)
    for blob_hash in _index_attachment_blob_hashes(index_db) if index_db is not None else ():
        inventory.setdefault(blob_hash, set()).add("index_attachment")
    hashes = set(inventory)
    store = BlobStore(source_blob_root)
    debt_report = scan_blob_reference_debt(
        source_db,
        store=store,
        sample_size=_MISSING_BLOB_WARNING_SAMPLE_LIMIT,
        immutable=True,
    )
    if not hashes:
        return 0, 0, debt_report

    blob_dst_root = backup_root / "blob"
    count = 0
    size = 0
    copied_inventory: list[dict[str, object]] = []
    missing_reserved: list[str] = []
    missing_index_attachments: list[str] = []
    for hash_hex in sorted(hashes):
        src = store.blob_path(hash_hex)
        if not src.exists():
            if "reserved" in inventory[hash_hex]:
                missing_reserved.append(hash_hex)
            if "index_attachment" in inventory[hash_hex]:
                missing_index_attachments.append(hash_hex)
            continue
        dst = blob_dst_root / hash_hex[:2] / hash_hex[2:]
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
            "source-tier publication receipts missing blob bytes: "
            f"{len(missing_reserved)} total"
            + (
                f" (sample: {', '.join(missing_reserved[:_MISSING_BLOB_WARNING_SAMPLE_LIMIT])})"
                if missing_reserved
                else ""
            )
        )
    if missing_index_attachments:
        warnings.append(
            "index-tier attachment references missing blob bytes: "
            f"{len(missing_index_attachments)} total"
            + (
                f" (sample: {', '.join(missing_index_attachments[:_MISSING_BLOB_WARNING_SAMPLE_LIMIT])})"
                if missing_index_attachments
                else ""
            )
        )
    if debt_report.missing_referenced_blobs:
        _write_blob_reference_debt_report(backup_root, debt_report)
        sample = ", ".join(debt_report.sample)
        warnings.append(
            "source-tier referenced blobs missing: "
            f"{debt_report.missing_referenced_blobs} total"
            + (f" (sample: {sample})" if sample else "")
            + "; details: blob-reference-debt.json"
            + " (this counts source.db/raw_sessions references only -- unfetched"
            " index-tier attachments with a NULL blob_hash are never counted"
            " here; see `polylogue ops maintenance attachment-acquisition-debt`"
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
    tier_source_fingerprints: dict[str, dict[str, object]],
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
        "tier_source_fingerprints": tier_source_fingerprints,
        "warnings": warnings,
    }
    if blob_reference_debt is not None:
        manifest["blob_reference_debt"] = blob_reference_debt.to_dict()
    (backup_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def backup_archive(
    *,
    output_dir: Path,
    check_only: bool = False,
    verify: bool = False,
    profile: BackupProfile = "rebuildable_cache_exclude",
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
    """
    started = time.monotonic()

    if check_only:
        warnings = _check_prerequisites(profile=profile)
        return BackupResult(
            ok=len(warnings) == 0,
            check_only=True,
            backup_mode="archive_file_set",
            backup_profile=profile,
            warnings=warnings,
            error=warnings[0] if warnings else None,
            elapsed_s=round(time.monotonic() - started, 3),
        )

    # Non-check mode: actually create backup.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = _backup_archive(output_dir=output_dir, started=started, profile=profile)
    if verify and result.ok and result.output_path is not None:
        _verify_backup_result(result)
    return result


def _backup_archive(*, output_dir: Path, started: float, profile: BackupProfile) -> BackupResult:
    root = archive_root()
    included_tiers = {
        tier: path
        for tier, path in _profile_archive_tiers(root, profile).items()
        if path.exists() or tier not in _optional_profile_tiers(profile)
    }
    omitted_tiers = {tier: path for tier, path in _all_archive_tiers(root).items() if tier not in included_tiers}
    warnings = _check_prerequisites(profile=profile)
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
            copied_size, fingerprint = _backup_sqlite(src, dst)
            db_size += copied_size
            tier_source_fingerprints[f"{tier}.db"] = fingerprint
            backed_up_files.append(str(dst))

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
    if blob_count:
        backed_up_files.append(str(backup_root / "blob"))

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
        tier_source_fingerprints=tier_source_fingerprints,
        blob_reference_debt=blob_reference_debt,
    )
    backed_up_files.append(str(backup_root / "manifest.json"))
    if (backup_root / "blob-inventory.json").exists():
        backed_up_files.append(str(backup_root / "blob-inventory.json"))
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
    conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
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
    env_tmpdir = os.environ.get("POLYLOGUE_BACKUP_VERIFY_TMPDIR")
    candidates = (path.parent, Path(env_tmpdir) if env_tmpdir else None, Path("/realm/tmp"))
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
    restore_root = scratch_root / "restore"
    shutil.copytree(source, restore_root, symlinks=True)
    return restore_root


def _remove_verification_receipt(backup_root: Path) -> None:
    (backup_root / _VERIFICATION_RECEIPT_FILE).unlink(missing_ok=True)


def _verify_archive_file_set_backup(path: Path) -> dict[str, object]:
    scratch_parent = _backup_verification_scratch_parent(path)
    with tempfile.TemporaryDirectory(prefix="polylogue-backup-verify-", dir=scratch_parent) as raw_tmp:
        restored = _copy_backup_artifact_to_scratch(path, Path(raw_tmp))
        if not restored.is_dir():
            return {"ok": False, "mode": "archive_file_set", "error": "backup output is not a directory"}

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
        expected_blobs = {
            str(item["blob_hash"]): int(item["size_bytes"])
            for item in inventory
            if isinstance(item, dict) and "blob_hash" in item and "size_bytes" in item
        }
        restored_blob_paths = _regular_backup_blob_files(restored)
        restored_blob_count = len(restored_blob_paths)
        restored_hashes: dict[str, int] = {}
        hashes_valid = True
        for blob_path in restored_blob_paths:
            blob_hash = blob_path.parent.name + blob_path.name
            payload = blob_path.read_bytes()
            restored_hashes[blob_hash] = len(payload)
            hashes_valid = hashes_valid and hashlib.sha256(payload).hexdigest() == blob_hash
        blobs_ok = (
            restored_blob_count == blob_count
            and len(expected_blobs) == blob_count
            and restored_hashes == expected_blobs
            and hashes_valid
        )
        restored_hash_set = set(restored_hashes)
        source_blob_hashes = _source_blob_hashes(restored / "source.db") if (restored / "source.db").exists() else set()
        missing_source_blobs = source_blob_hashes - restored_hash_set
        source_blobs_resolved = not missing_source_blobs
        index_attachment_hashes = _index_attachment_blob_hashes(restored / "index.db")
        missing_index_attachment_blobs = index_attachment_hashes - restored_hash_set
        index_attachment_blobs_resolved = not missing_index_attachment_blobs
        ok = (
            all(tier_integrity.values())
            and omitted_absent
            and blobs_ok
            and source_blobs_resolved
            and index_attachment_blobs_resolved
        )
        receipt_evidence = _receipt_evidence(restored) if ok else None
        return {
            "ok": ok,
            "mode": "archive_file_set",
            "profile": manifest.get("profile", "rebuildable_cache_exclude"),
            "tier_integrity": tier_integrity,
            "omitted_tiers_absent": omitted_absent,
            "manifest_blob_count": blob_count,
            "restored_blob_count": restored_blob_count,
            "blob_inventory_exact": blobs_ok,
            "source_blobs_resolved": source_blobs_resolved,
            "missing_source_blob_count": len(missing_source_blobs),
            "index_attachment_blobs_resolved": index_attachment_blobs_resolved,
            "missing_index_attachment_blob_count": len(missing_index_attachment_blobs),
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
        if not isinstance(source_fingerprint, dict) or any(
            artifact[field] != source_fingerprint.get(field) for field in ("size_bytes", "sha256", "user_version")
        ):
            raise RuntimeError(f"{filename} backup artifact does not match its live source fingerprint")
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


def _receipt_evidence(backup_root: Path) -> dict[str, object]:
    _require_real_backup_directory(backup_root, label="backup root")
    artifact_inventory = _backup_artifact_inventory(backup_root)
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
            if not isinstance(artifact, dict) or artifact.get("tier") not in {"source", "user"}:
                continue
            fingerprint = artifact.get("source_fingerprint")
            source_path = fingerprint.get("path") if isinstance(fingerprint, dict) else None
            if isinstance(source_path, str) and source_path:
                authority_paths[str(artifact["tier"])] = Path(source_path).resolve(strict=False)
    receipt = sign_verification_receipt(receipt_body, authority_paths=authority_paths)
    receipt_path = backup_root / _VERIFICATION_RECEIPT_FILE
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=backup_root,
        prefix=f".{_VERIFICATION_RECEIPT_FILE}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    try:
        os.replace(temporary_path, receipt_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return receipt_path


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
    "format_backup_result",
]

"""Backup and portability operations for the Polylogue archive.

Provides a local-first backup command for tiered archives.
Backups copy only the precious tiers plus referenced blobs: source.db, user.db,
embeddings.db, and blob files. Rebuildable index.db and disposable ops.db are
omitted by design.

Uses SQLite VACUUM INTO when available (SQLite >= 3.27.0) for a clean,
defragmented copy. Falls back to a file copy with WAL checkpoint first.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from polylogue.logging import get_logger
from polylogue.paths import archive_root, blob_store_root
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


def _backup_db_vacuum_into(src: Path, dst: Path) -> None:
    """Backup using VACUUM INTO (SQLite >= 3.27.0)."""
    conn = sqlite3.connect(str(src))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM INTO ?", (str(dst),))
    finally:
        conn.close()


def _backup_db_copy(src: Path, dst: Path) -> None:
    """Backup using file copy after WAL checkpoint."""
    conn = sqlite3.connect(str(src))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    shutil.copy2(src, dst)


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

    # Check available disk space (rough estimate: 2x db size for VACUUM INTO).
    try:
        db_size = 0
        for path in _profile_archive_tiers(root, profile).values():
            if path.exists():
                db_size += path.stat().st_size
                wal = path.with_suffix(".db-wal")
                if wal.exists():
                    db_size += wal.stat().st_size
        # Estimate need ~2.5x for VACUUM INTO overhead.
        needed = int(db_size * 2.5)
        import os

        st = os.statvfs(str(root))
        free = st.f_frsize * st.f_bavail
        if free < needed:
            warnings.append(
                f"low disk space: {free / (1024**3):.1f} GB free, ~{needed / (1024**3):.1f} GB needed for VACUUM INTO"
            )
    except Exception:
        pass

    return warnings


def _has_backup_error(warnings: list[str]) -> bool:
    return any("not found" in warning or "not readable" in warning for warning in warnings)


def _backup_sqlite(src: Path, dst: Path) -> int:
    try:
        _backup_db_vacuum_into(src, dst)
    except sqlite3.OperationalError:
        _backup_db_copy(src, dst)
    return dst.stat().st_size if dst.exists() else 0


def _source_blob_hashes(source_db: Path) -> set[str]:
    conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT DISTINCT hex(blob_hash) FROM blob_refs").fetchall()
    finally:
        conn.close()
    return {str(row[0]).lower() for row in rows if row and row[0]}


def _copy_referenced_blobs(*, source_db: Path, backup_root: Path, warnings: list[str]) -> tuple[int, int]:
    hashes = _source_blob_hashes(source_db)
    if not hashes:
        return 0, 0

    store = BlobStore(blob_store_root())
    blob_dst_root = backup_root / "blob"
    count = 0
    size = 0
    missing_count = 0
    missing_samples: list[str] = []
    for hash_hex in sorted(hashes):
        src = store.blob_path(hash_hex)
        if not src.exists():
            missing_count += 1
            if len(missing_samples) < _MISSING_BLOB_WARNING_SAMPLE_LIMIT:
                missing_samples.append(hash_hex)
            continue
        dst = blob_dst_root / hash_hex[:2] / hash_hex[2:]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        count += 1
        size += dst.stat().st_size
    if missing_count:
        warnings.append(
            "referenced blobs missing: "
            f"{missing_count} total" + (f" (sample: {', '.join(missing_samples)})" if missing_samples else "")
        )
    return count, size


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
        "warnings": warnings,
    }
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
    for tier, src in included_tiers.items():
        dst = backup_root / f"{tier}.db"
        db_size += _backup_sqlite(src, dst)
        backed_up_files.append(str(dst))

    if "source" in included_tiers:
        blob_count, blob_size = _copy_referenced_blobs(
            source_db=included_tiers["source"],
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
    )
    backed_up_files.append(str(backup_root / "manifest.json"))

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
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
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
    try:
        verification = _verify_archive_file_set_backup(output_path)
    except Exception as exc:
        verification = {"ok": False, "error": str(exc)}

    result.verification = verification
    result.verified = bool(verification.get("ok"))
    if not result.verified:
        result.ok = False
        result.error = str(verification.get("error") or "backup verification failed")


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
    restore_root = scratch_root / "restore"
    if source.is_dir():
        shutil.copytree(source, restore_root, symlinks=True)
        return restore_root
    restore_root.mkdir()
    shutil.copy2(source, restore_root / source.name)
    return restore_root / source.name


def _verify_archive_file_set_backup(path: Path) -> dict[str, object]:
    scratch_parent = _backup_verification_scratch_parent(path)
    with tempfile.TemporaryDirectory(prefix="polylogue-backup-verify-", dir=scratch_parent) as raw_tmp:
        restored = _copy_backup_artifact_to_scratch(path, Path(raw_tmp))
        if not restored.is_dir():
            return {"ok": False, "mode": "archive_file_set", "error": "backup output is not a directory"}

        manifest_path = restored / "manifest.json"
        if not manifest_path.exists():
            return {"ok": False, "mode": "archive_file_set", "error": "manifest.json is missing"}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        included_tiers = [
            str(item) for item in manifest.get("included_tiers", ("source.db", "user.db", "embeddings.db"))
        ]
        omitted_tiers = [str(item) for item in manifest.get("omitted_tiers", ("index.db", "ops.db"))]
        tier_integrity = {
            name.removesuffix(".db"): _sqlite_integrity_ok(restored / name) if (restored / name).exists() else False
            for name in included_tiers
            if name.endswith(".db")
        }
        omitted_absent = all(not (restored / name).exists() for name in omitted_tiers)
        blob_count = int(manifest.get("blob_count", 0) or 0)
        restored_blob_count = sum(1 for path_ in (restored / "blob").rglob("*") if path_.is_file()) if blob_count else 0
        blobs_ok = restored_blob_count == blob_count
        ok = all(tier_integrity.values()) and omitted_absent and blobs_ok
        return {
            "ok": ok,
            "mode": "archive_file_set",
            "profile": manifest.get("profile", "rebuildable_cache_exclude"),
            "tier_integrity": tier_integrity,
            "omitted_tiers_absent": omitted_absent,
            "manifest_blob_count": blob_count,
            "restored_blob_count": restored_blob_count,
            "scratch_restore": "temporary",
            "scratch_parent": str(Path(raw_tmp).parent),
        }


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

"""Backup and portability operations for the Polylogue archive.

Provides a local-first backup command for tiered archives.
Backups copy the authority and precious tiers plus referenced blobs: audit.db,
source.db, user.db, embeddings.db, and blob files. Rebuildable index.db and
disposable ops.db are omitted by profiles that do not request full evidence.

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
from collections.abc import Mapping
from contextlib import AbstractContextManager, closing, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from polylogue.core.durable_fs import atomic_replace
from polylogue.logging import get_logger
from polylogue.operations.zip_acquisition_replay import zip_reacquisition_payload
from polylogue.paths import archive_root
from polylogue.storage.backup_attestation import (
    VERIFICATION_RECEIPT_FORMAT,
    archive_tier_paths,
    sign_verification_receipt,
)
from polylogue.storage.blob_integrity import (
    BlobLivenessProjection,
    BlobReferenceDebtReport,
    _current_raw_payload_bytes,
    _raw_session_reference_rows,
    blob_reference_debt_from_projection,
    project_source_blob_liveness,
)
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
_BLOB_REFERENCE_EVIDENCE_FILE = "blob-reference-evidence.json"
_SOURCE_DECLARED_ABSENT_FILE = "source-declared-absent.json"
_SOURCE_DECLARED_ABSENT_FORMAT = "polylogue-source-declared-absent-v1"
_SOURCE_DECLARED_ABSENT_AUTHORITY = "polylogue-2x6xu"
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
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sqlite_user_version(path: Path) -> int:
    with closing(sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)


def _readable_sqlite_index(path: Path) -> bool:
    """Return whether an active-pointer candidate is a readable SQLite index.

    Backup follows a genuinely live external index target, but a malformed or
    stale pointer must not turn an otherwise valid backup into a copy of
    arbitrary bytes. Relocation still authenticates that pointer separately.
    """
    try:
        _sqlite_user_version(path)
    except (OSError, sqlite3.Error):
        return False
    return True


def _sqlite_source_fingerprint(path: Path) -> dict[str, object]:
    metadata = path.stat()
    return {
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "size_bytes": metadata.st_size,
        "sha256": _sha256_file(path),
        "user_version": _sqlite_user_version(path),
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


def _source_blob_liveness_projection(
    source_db: Path, *, index_db: Path | None = None, source_generation_id: str | None = None
) -> tuple[BlobLivenessProjection, set[str]]:
    """Read complete source evidence or refuse the backup before copying blobs."""

    projection = project_source_blob_liveness(
        source_db,
        index_db=index_db,
        immutable=True,
        source_generation_id=source_generation_id,
    )
    return projection, _source_blob_reservations(source_db)


def _latest_sealed_source_generation(source_db: Path) -> str | None:
    """Return the newest complete source generation, if this tier has them."""
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro&immutable=1", uri=True)) as conn:
        if not _source_generation_tables_exist(conn):
            return None
        row = conn.execute(
            """SELECT g.source_generation_id
               FROM source_generations AS g
              WHERE g.sealed_at_ms IS NOT NULL
                AND NOT EXISTS (
                    SELECT 1 FROM source_item_reconciliation AS r
                     WHERE r.source_generation_id = g.source_generation_id
                       AND NOT r.sealable
                )
              ORDER BY g.sealed_at_ms DESC, g.source_generation_id DESC
              LIMIT 1"""
        ).fetchone()
        return str(row[0]) if row is not None else None


def _source_generation_tables_exist(conn: sqlite3.Connection) -> bool:
    """Return whether the source tier has crossed the generation migration."""

    tables = conn.execute(
        "SELECT name FROM sqlite_schema WHERE type='table' AND name IN ('source_generations', 'source_items')"
    ).fetchall()
    return len(tables) == 2


def _source_blob_reservations(source_db: Path) -> set[str]:
    """Read pending publication receipts independently of committed liveness."""

    with closing(sqlite3.connect(f"file:{source_db}?mode=ro&immutable=1", uri=True)) as source_conn:
        has_reservations = source_conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'blob_publication_reservations'"
        ).fetchone()
        if has_reservations is None:
            return set()
        columns = {str(row[1]) for row in source_conn.execute("PRAGMA table_info(blob_publication_reservations)")}
        if "blob_hash" not in columns:
            raise RuntimeError("source.blob_publication_reservations is missing columns: blob_hash")
        reservations: set[str] = set()
        for (blob_hash,) in source_conn.execute("SELECT DISTINCT blob_hash FROM blob_publication_reservations"):
            if not isinstance(blob_hash, bytes) or len(blob_hash) != 32:
                raise RuntimeError("source.blob_publication_reservations has invalid blob_hash evidence")
            reservations.add(blob_hash.hex())
        return reservations


def _source_blob_hashes_from_restored_source(source_db: Path, *, source_generation_id: str | None) -> set[str]:
    """Re-derive source-owned hashes from the restored source tier."""

    projection = project_source_blob_liveness(
        source_db,
        immutable=True,
        source_generation_id=source_generation_id,
    )
    if projection.blockers:
        raise RuntimeError("restored source blob reference projection is blocked: " + "; ".join(projection.blockers))
    return set(projection.live_hashes)


def _load_source_declared_absent(source_db: Path, assertion_path: Path) -> set[str]:
    """Load and authenticate the operator declaration against ``source.db``."""

    _require_regular_backup_artifact(
        assertion_path,
        backup_root=assertion_path.parent,
        label="source declared-absent assertion",
    )
    try:
        assertion = json.loads(assertion_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("source declared-absent assertion is not valid JSON") from exc
    if not isinstance(assertion, dict) or assertion.get("format") != _SOURCE_DECLARED_ABSENT_FORMAT:
        raise RuntimeError("source declared-absent assertion has an unknown format")
    if assertion.get("freeze_authority") != _SOURCE_DECLARED_ABSENT_AUTHORITY:
        raise RuntimeError("source declared-absent assertion lacks polylogue-2x6xu freeze authority")
    if assertion.get("source_db_sha256") != _sha256_file(source_db):
        raise RuntimeError("source declared-absent assertion is bound to different source.db bytes")
    raw_hashes = assertion.get("declared_absent_blob_hashes")
    if not isinstance(raw_hashes, list) or not raw_hashes:
        raise RuntimeError("source declared-absent assertion has an empty declared set")
    if any(
        not isinstance(blob_hash, str)
        or len(blob_hash) != 64
        or any(char not in "0123456789abcdef" for char in blob_hash)
        for blob_hash in raw_hashes
    ):
        raise RuntimeError("source declared-absent assertion has invalid blob hashes")
    hashes = {str(blob_hash) for blob_hash in raw_hashes}
    if len(hashes) != len(raw_hashes):
        raise RuntimeError("source declared-absent assertion contains duplicate blob hashes")
    return hashes


def _copy_source_declared_absent_assertion(source_db: Path, backup_root: Path) -> Path | None:
    """Copy the optional durable source assertion into a backup package."""

    source_path = source_db.with_name(_SOURCE_DECLARED_ABSENT_FILE)
    if not source_path.exists() and not source_path.is_symlink():
        return None
    _require_regular_backup_artifact(
        source_path, backup_root=source_db.parent, label="source declared-absent assertion"
    )
    destination = backup_root / _SOURCE_DECLARED_ABSENT_FILE
    shutil.copy2(source_path, destination)
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
        with closing(sqlite3.connect(f"file:{index_db}?mode=ro&immutable=1", uri=True)) as index_conn:
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
    source_generation_id: str | None = None,
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
        "source_generation_id": source_generation_id,
        "source_owner_hashes": source_owners,
        "index_attachment_evidence": attachment_evidence_state,
        "index_attachment_hashes": sorted(attachment_hashes),
    }


def _resolved_source_path(source_path: str, root: Path) -> str:
    """Resolve an acquisition path against the archive root in force."""
    outer, separator, member = source_path.partition(":")
    path = Path(outer)
    if not path.exists():
        parts = path.parts
        for directory in ("inbox", "browser-capture", "hooks"):
            if directory in parts:
                candidate = root.joinpath(*parts[parts.index(directory) :])
                if candidate.exists():
                    path = candidate
                    break
    return f"{path}:{member}" if separator else str(path)


def _source_recoverability_proofs(
    source_db: Path,
    *,
    root: Path,
    missing_hashes: set[str],
    source_bytes_cache: dict[str, bytes] | None = None,
    decoded_payload_cache: dict[str, object] | None = None,
    zip_payload_cache: dict[str, dict[int, bytes]] | None = None,
) -> list[dict[str, str]]:
    """Prove missing source-owned bytes by replaying their acquisition payload."""
    if not missing_hashes:
        return []
    source_bytes_cache = source_bytes_cache if source_bytes_cache is not None else {}
    decoded_payload_cache = decoded_payload_cache if decoded_payload_cache is not None else {}
    zip_payload_cache = zip_payload_cache if zip_payload_cache is not None else {}
    by_hash: dict[str, list[dict[str, object]]] = {}
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro&immutable=1", uri=True)) as conn:
        for row in _raw_session_reference_rows(conn):
            blob_hash = str(row.get("blob_hash") or "")
            if blob_hash in missing_hashes:
                by_hash.setdefault(blob_hash, []).append(row)
    proofs: list[dict[str, str]] = []
    for blob_hash, rows in by_hash.items():
        for row in rows:
            source_path = row.get("source_path")
            if not isinstance(source_path, str) or not source_path:
                continue
            source_index_value = row.get("source_index")
            source_index = int(source_index_value) if isinstance(source_index_value, (int, str)) else None
            resolved = _resolved_source_path(source_path, root)
            if ":" in resolved:
                payload, error = zip_reacquisition_payload(
                    row,
                    source_path=resolved,
                    zip_payload_cache=zip_payload_cache,
                )
            else:
                payload, error = _current_raw_payload_bytes(
                    resolved,
                    source_index,
                    raw_id=str(row.get("ref_id") or "") or None,
                    blob_hash=blob_hash,
                    source_bytes_cache=source_bytes_cache,
                    decoded_payload_cache=decoded_payload_cache,
                )
            if error is None and payload is not None and hashlib.sha256(payload).hexdigest() == blob_hash:
                kind = "zip_reacquired_payload" if ":" in resolved else "direct_file_sha256"
                proofs.append(
                    {
                        "blob_hash": blob_hash,
                        "kind": kind,
                        "source_path": resolved,
                        "raw_id": str(row.get("ref_id") or ""),
                        "source_index": str(row.get("source_index")) if row.get("source_index") is not None else "",
                        "origin": str(row.get("origin") or ""),
                        "capture_mode": str(row.get("capture_mode") or ""),
                        "coordinate_format": str(row.get("coordinate_format") or ""),
                        "entry_ordinal": str(row.get("entry_ordinal")) if row.get("entry_ordinal") is not None else "",
                        "split_index": str(row.get("split_index")) if row.get("split_index") is not None else "",
                    }
                )
                break
    return proofs


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
    source_generation_id: str | None = None,
) -> tuple[int, int, BlobReferenceDebtReport]:
    if source_generation_id is None:
        projection, reservations = _source_blob_liveness_projection(source_db, index_db=index_db)
    else:
        projection, reservations = _source_blob_liveness_projection(
            source_db, index_db=index_db, source_generation_id=source_generation_id
        )
    reference_evidence = _blob_reference_evidence(
        projection, index_db=index_db, source_generation_id=source_generation_id
    )
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
    reference_evidence["recoverability_proofs"] = _source_recoverability_proofs(
        source_db,
        root=source_db.parent,
        missing_hashes=missing_hashes & source_hashes,
        source_bytes_cache={},
        decoded_payload_cache={},
        zip_payload_cache={},
    )
    _write_blob_reference_evidence(backup_root, reference_evidence)
    if not hashes:
        return 0, 0, debt_report

    blob_dst_root = backup_root / "blob"
    count = 0
    size = 0
    copied_inventory: list[dict[str, object]] = []
    missing_reserved: list[str] = []
    for hash_hex in sorted(hashes):
        src = store.blob_path(hash_hex)
        if not src.exists():
            if inventory[hash_hex] == {"reserved"}:
                missing_reserved.append(hash_hex)
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
            "source-tier publication reservations missing blob bytes: "
            f"{len(missing_reserved)} total"
            + (
                f" (sample: {', '.join(missing_reserved[:_MISSING_BLOB_WARNING_SAMPLE_LIMIT])})"
                if missing_reserved
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
    blob_reference_debt: BlobReferenceDebtReport | None = None,
    source_generation_id: str | None = None,
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
        "warnings": warnings,
        "source_generation_id": source_generation_id,
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

        source_assertion = (
            _copy_source_declared_absent_assertion(root / "source.db", backup_root)
            if "source" in included_tiers
            else None
        )

        blob_reference_debt: BlobReferenceDebtReport | None = None
        source_generation_id = (
            _latest_sealed_source_generation(backup_root / "source.db") if "source" in included_tiers else None
        )
        if "source" in included_tiers:
            blob_count, blob_size, blob_reference_debt = _copy_referenced_blobs(
                source_db=backup_root / "source.db",
                source_blob_root=root / "blob",
                index_db=(backup_root / "index.db" if "index" in included_tiers else None),
                backup_root=backup_root,
                warnings=warnings,
                source_generation_id=source_generation_id,
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
        archive_root_source_identity=_archive_root_source_identity(root),
        tier_source_fingerprints=tier_source_fingerprints,
        blob_reference_debt=blob_reference_debt,
        source_generation_id=source_generation_id,
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
            payload = blob_path.read_bytes()
            restored_hashes[blob_hash] = len(payload)
            payload_hash = hashlib.sha256(payload).hexdigest()
            hashes_valid = hashes_valid and payload_hash == blob_hash
            verified_blob_file_hashes[str(blob_path.relative_to(restored))] = (len(payload), payload_hash)
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
        recoverable_source_hashes: set[str] = set()
        if source_included:
            evidence_path = restored / str(manifest.get("blob_reference_evidence_file", _BLOB_REFERENCE_EVIDENCE_FILE))
            if not evidence_path.exists() and not evidence_path.is_symlink():
                raise RuntimeError("backup blob reference evidence is missing")
            _require_regular_backup_artifact(
                evidence_path, backup_root=restored, label="backup blob reference evidence"
            )
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            expected_reference_blobs = _expected_blob_hashes_from_evidence(evidence)
            source_evidence_hashes = {
                blob_hash for hashes in evidence["source_owner_hashes"].values() for blob_hash in hashes
            }
            recoverability_proofs = evidence.get("recoverability_proofs", [])
            if not isinstance(recoverability_proofs, list):
                reference_evidence_ok = False
                recoverability_proofs = []
            source_bytes_cache: dict[str, bytes] = {}
            decoded_payload_cache: dict[str, object] = {}
            zip_payload_cache: dict[str, dict[int, bytes]] = {}
            for proof in recoverability_proofs:
                if not isinstance(proof, dict):
                    reference_evidence_ok = False
                    continue
                blob_hash_value = proof.get("blob_hash")
                source_path_value = proof.get("source_path")
                kind = proof.get("kind")
                if (
                    not isinstance(blob_hash_value, str)
                    or blob_hash_value not in source_evidence_hashes
                    or not isinstance(source_path_value, str)
                    or kind not in {"direct_file_sha256", "zip_reacquired_payload"}
                ):
                    reference_evidence_ok = False
                    continue
                blob_hash = blob_hash_value
                source_path = source_path_value
                source_index_value = proof.get("source_index")
                source_index = (
                    int(source_index_value) if isinstance(source_index_value, str) and source_index_value else None
                )
                if kind == "zip_reacquired_payload":
                    recovered_payload, recovery_error = zip_reacquisition_payload(
                        proof,
                        source_path=source_path,
                        zip_payload_cache=zip_payload_cache,
                    )
                else:
                    recovered_payload, recovery_error = _current_raw_payload_bytes(
                        source_path,
                        source_index,
                        raw_id=str(proof.get("raw_id") or "") or None,
                        blob_hash=blob_hash,
                        source_bytes_cache=source_bytes_cache,
                        decoded_payload_cache=decoded_payload_cache,
                    )
                payload_matches = (
                    recovered_payload is not None and hashlib.sha256(recovered_payload).hexdigest() == blob_hash
                )
                if recovery_error is not None or not payload_matches:
                    reference_evidence_ok = False
                else:
                    recoverable_source_hashes.add(blob_hash)
            expected_attachment_hashes = set(evidence["index_attachment_hashes"])
            if evidence["index_attachment_evidence"] == "consulted":
                if not index_path.exists():
                    raise RuntimeError("backup index attachment evidence was consulted but index.db is missing")
                observed_attachment_hashes = _index_attachment_hashes(index_path)
                reference_evidence_ok = (
                    reference_evidence_ok and observed_attachment_hashes == expected_attachment_hashes
                )
            source_generation_id = evidence.get("source_generation_id")
            if source_generation_id is not None and not isinstance(source_generation_id, str):
                raise RuntimeError("backup blob reference evidence has invalid source generation identity")
            with closing(
                sqlite3.connect(f"file:{restored / 'source.db'}?mode=ro&immutable=1", uri=True)
            ) as source_conn:
                source_generation_tables_exist = _source_generation_tables_exist(source_conn)
            restored_source_hashes = _source_blob_hashes_from_restored_source(
                restored / "source.db",
                source_generation_id=source_generation_id,
            )
            reference_evidence_ok = reference_evidence_ok and restored_source_hashes == source_evidence_hashes
            assertion_path = restored / _SOURCE_DECLARED_ABSENT_FILE
            declared_absent: set[str] = set()
            if assertion_path.exists() or assertion_path.is_symlink():
                if source_generation_id is not None or source_generation_tables_exist:
                    raise RuntimeError("source declared-absent assertion is only valid before source generations exist")
                declared_absent = _load_source_declared_absent(restored / "source.db", assertion_path)
                if not declared_absent.issubset(restored_source_hashes):
                    reference_evidence_ok = False
                effective_source_hashes = restored_source_hashes - declared_absent - recoverable_source_hashes
            else:
                effective_source_hashes = restored_source_hashes - recoverable_source_hashes
            reservations = _source_blob_reservations(restored / "source.db")
            expected_reference_blobs = effective_source_hashes | expected_attachment_hashes | reservations
            if assertion_path.exists() or assertion_path.is_symlink():
                source_scope_ok = bool(effective_source_hashes)
        missing_canonical_blobs = expected_reference_blobs - restored_hash_set
        canonical_blobs_resolved = not source_included or (
            not missing_canonical_blobs and reference_evidence_ok and source_scope_ok
        )
        ok = all(tier_integrity.values()) and omitted_absent and blobs_ok and canonical_blobs_resolved
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
            "recoverable_source_blob_count": len(recoverable_source_hashes),
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
    atomic_replace(receipt_path, json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8"))
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

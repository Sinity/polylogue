"""Backup-gated reconciliation of one live cursor-authority violation.

The command is intentionally narrow.  It proves one source path from the
canonical raw-frontier projection, then runs that path through
``LiveBatchProcessor.ingest_files``.  It never edits a cursor or accepted head
itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import tempfile
import time
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path

from polylogue.api import Polylogue
from polylogue.config import Config
from polylogue.operations.durable_change_train import acquire_durable_archive_ownership
from polylogue.sources.live.batch import (
    LiveBatchProcessor,
    cursor_authority_path_digest,
    scoped_cursor_authority_authorization,
)
from polylogue.sources.live.batch_support import sha256_range_from_path
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.metrics import LiveBatchMetrics
from polylogue.sources.live.watcher import WatchSource
from polylogue.storage.raw_retention import RawFrontierIntegrityProjection, raw_frontier_integrity_projection

PLAN_FORMAT = "polylogue.cursor-authority-reconciliation-plan.v1"
RECEIPT_FORMAT = "polylogue.cursor-authority-reconciliation-receipt.v1"
ARCHIVE_ROOT = Path("/realm/db/polylogue")
_REQUIRED_TIERS = ("source", "index", "ops", "audit")


class CursorAuthorityReconciliationError(RuntimeError):
    """A reconciliation precondition or postcondition was not proven."""


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> tuple[int, str]:
    """Return size and digest from one descriptor observation."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CursorAuthorityReconciliationError(f"required archive file is unreadable: {path}") from exc
    return size, digest.hexdigest()


def _stat_observation(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _archive_root() -> Path:
    """Return the fixed archive root for this command.

    This deliberately does not call ``polylogue.paths.archive_root`` or read
    any ambient archive-root environment variable.
    """

    return ARCHIVE_ROOT


def _read_private_source_path(path_file: Path) -> Path:
    descriptor: int | None = None
    try:
        descriptor = os.open(path_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise CursorAuthorityReconciliationError(f"source path file is unreadable: {path_file}") from exc
    try:
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CursorAuthorityReconciliationError("source path file must be a regular single-linked file")
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise CursorAuthorityReconciliationError("source path file must be owned by the operator and mode 0600")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError as exc:
        raise CursorAuthorityReconciliationError(f"source path file is unreadable: {path_file}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        lines = b"".join(chunks).decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise CursorAuthorityReconciliationError("source path file must be UTF-8 text") from exc
    if len(lines) != 1 or not lines[0].strip():
        raise CursorAuthorityReconciliationError("source path file must contain exactly one non-empty path")
    candidate = Path(lines[0])
    if not candidate.is_absolute():
        raise CursorAuthorityReconciliationError("selected source path must be absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CursorAuthorityReconciliationError("selected source path does not resolve") from exc
    if not resolved.is_file():
        raise CursorAuthorityReconciliationError("selected source path must be a regular file")
    return resolved


def _sqlite_snapshot(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise CursorAuthorityReconciliationError(f"required archive tier is missing: {path}")
    size_bytes, sha256 = _file_fingerprint(path)
    try:
        with closing(sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)) as conn:
            conn.execute("PRAGMA query_only = ON")
            user_version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
            schema_version = int(conn.execute("PRAGMA schema_version").fetchone()[0] or 0)
            schema_rows = conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_schema "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name, tbl_name"
            ).fetchall()
            quick_check = tuple(str(row[0]) for row in conn.execute("PRAGMA quick_check"))
    except (OSError, sqlite3.Error) as exc:
        raise CursorAuthorityReconciliationError(f"could not read SQLite tier {path}: {exc}") from exc
    schema_digest = _canonical_digest(
        [[str(value) if value is not None else None for value in row] for row in schema_rows]
    )
    return {
        "size_bytes": size_bytes,
        "sha256": sha256,
        "user_version": user_version,
        "schema_version": schema_version,
        "schema_sha256": schema_digest,
        "quick_check": list(quick_check),
    }


def _tier_snapshots(root: Path) -> dict[str, dict[str, object]]:
    snapshots: dict[str, dict[str, object]] = {}
    for tier in _REQUIRED_TIERS:
        snapshots[tier] = _sqlite_snapshot(root / f"{tier}.db")
    return snapshots


def _code_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _deployed_package_sha() -> str:
    package_root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        if any(part == "__pycache__" for part in path.parts):
            continue
        digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _projection_for(root: Path) -> RawFrontierIntegrityProjection:
    from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot

    return raw_frontier_integrity_projection(
        root,
        raw_materialization_readiness_snapshot(root),
        sample_limit=100,
    )


def _private_projection(projection: RawFrontierIntegrityProjection) -> dict[str, object]:
    def redact(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: (
                    cursor_authority_path_digest(Path(str(item)))
                    if key in {"source_path", "logical_source_key"} and isinstance(item, str)
                    else None
                    if key in {"source_path", "logical_source_key"} and item is not None
                    else redact(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    redacted = redact(projection.to_dict())
    if not isinstance(redacted, dict):
        raise CursorAuthorityReconciliationError("raw-frontier projection did not produce a mapping")
    return redacted


def _cursor_rows(root: Path) -> list[tuple[str, int]]:
    with closing(sqlite3.connect(f"file:{(root / 'ops.db').resolve()}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT source_path, byte_offset FROM ingest_cursor "
            "WHERE COALESCE(excluded, 0) = 0 AND byte_offset IS NOT NULL"
        ).fetchall()
    result: list[tuple[str, int]] = []
    for row in rows:
        if not isinstance(row[0], str) or not row[0]:
            raise CursorAuthorityReconciliationError("ingest cursor has an invalid source path")
        result.append((row[0], _required_nonnegative_int(row[1], "ingest cursor byte_offset")))
    return result


def _find_path_by_digest(root: Path, digest: str) -> Path:
    matches = [
        Path(path).resolve()
        for path, _offset in _cursor_rows(root)
        if cursor_authority_path_digest(Path(path)) == digest
    ]
    if len(matches) != 1:
        raise CursorAuthorityReconciliationError("plan path digest does not identify exactly one current cursor path")
    return matches[0]


def _head_details(root: Path, source_path: Path, projection: RawFrontierIntegrityProjection) -> dict[str, object]:
    if projection.cursor_ahead_count != 1 or len(projection.cursor_ahead_samples) != 1:
        raise CursorAuthorityReconciliationError("reconciliation requires exactly one true cursor-ahead row")
    sample = projection.cursor_ahead_samples[0]
    if Path(sample.source_path).resolve() != source_path.resolve():
        raise CursorAuthorityReconciliationError("selected source path is not the sole cursor-ahead path")
    with closing(sqlite3.connect(f"file:{(root / 'index.db').resolve()}?mode=ro", uri=True)) as conn:
        head = conn.execute(
            "SELECT logical_source_key, accepted_raw_id, accepted_source_revision, "
            "accepted_content_hash, accepted_frontier_kind, accepted_frontier, "
            "acquisition_generation, append_end_offset "
            "FROM raw_revision_heads WHERE logical_source_key = ?",
            (sample.logical_source_key,),
        ).fetchone()
    if head is None:
        raise CursorAuthorityReconciliationError("accepted head is missing for the selected path")
    with closing(sqlite3.connect(f"file:{(root / 'source.db').resolve()}?mode=ro", uri=True)) as conn:
        raw = conn.execute(
            "SELECT raw_id, source_path, blob_hash, blob_size, revision_authority FROM raw_sessions WHERE raw_id = ?",
            (str(head[1]),),
        ).fetchone()
    if raw is None or Path(str(raw[1])).resolve() != source_path.resolve():
        raise CursorAuthorityReconciliationError("accepted head does not match the recorded source path")
    if str(head[4]) != "byte" or str(raw[4]) != "byte_proven":
        raise CursorAuthorityReconciliationError("accepted head is not byte-authoritative")
    logical_source_key = head[0]
    if not isinstance(logical_source_key, str) or not logical_source_key:
        raise CursorAuthorityReconciliationError("accepted head has an invalid logical source key")
    frontier = _required_nonnegative_int(head[5], "accepted head frontier")
    blob_hash = bytes(raw[2]).hex() if isinstance(raw[2], bytes) else str(raw[2]).lower()
    blob_size = _required_nonnegative_int(raw[3], "accepted raw blob size")
    try:
        bytes.fromhex(blob_hash)
    except ValueError as exc:
        raise CursorAuthorityReconciliationError("accepted raw has an invalid blob hash") from exc
    if blob_size != frontier or len(blob_hash) != 64:
        raise CursorAuthorityReconciliationError("accepted raw does not bind a complete byte frontier")
    cursor_matches = [offset for path, offset in _cursor_rows(root) if Path(path).resolve() == source_path.resolve()]
    if len(cursor_matches) != 1:
        raise CursorAuthorityReconciliationError("selected source path has no unique current cursor row")
    cursor_offset = cursor_matches[0]
    before_stat = _stat_observation(source_path)
    prefix_digest, bytes_read = sha256_range_from_path(source_path, start_offset=0, end_offset=frontier)
    after_stat = _stat_observation(source_path)
    if before_stat != after_stat:
        raise CursorAuthorityReconciliationError("source mutated during accepted-frontier hashing")
    if prefix_digest != blob_hash:
        raise CursorAuthorityReconciliationError("source prefix does not match the accepted raw blob hash")
    return {
        "logical_source_key": cursor_authority_path_digest(Path(logical_source_key)),
        "cursor_byte_offset": cursor_offset,
        "accepted_frontier": frontier,
        "accepted_raw_id_digest": _canonical_digest(str(head[1])),
        "accepted_blob_hash_digest": _canonical_digest(blob_hash),
        "source_prefix_digest": prefix_digest,
        "source_prefix_bytes": bytes_read,
        "source_stat": list(after_stat),
    }


def _build_plan(root: Path, source_path: Path, *, require_candidate: bool = True) -> dict[str, object]:
    tiers = _tier_snapshots(root)
    projection = _projection_for(root)
    path_digest = cursor_authority_path_digest(source_path)
    if projection.cursor_ahead_count == 0:
        if require_candidate and projection.cursor_authority_gap_count == 0 and projection.overall_status == "healthy":
            not_applicable_plan: dict[str, object] = {
                "format": PLAN_FORMAT,
                "archive_identity": {"root": str(root.resolve())},
                "code_sha": _code_sha(),
                "deployed_package_sha": _deployed_package_sha(),
                "tier_fingerprints": tiers,
                "source_schema_versions": {tier: tiers[tier]["user_version"] for tier in _REQUIRED_TIERS},
                "selected_path_digest": path_digest,
                "observed_at_ms": int(time.time() * 1000),
                "status": "not_applicable",
                "cursor_byte_offset": None,
                "accepted_frontier": None,
                "accepted_raw_id_digest": None,
                "source_prefix_digest": None,
                "before_projection": _private_projection(projection),
            }
            not_applicable_plan["plan_digest"] = _canonical_digest(not_applicable_plan)
            return not_applicable_plan
        raise CursorAuthorityReconciliationError("cursor authority is incomparable or has no selected violation")
    if projection.cursor_ahead_count != 1:
        raise CursorAuthorityReconciliationError("refusing to guess among multiple cursor-ahead rows")
    if projection.broken_head_count or projection.missing_source_raw_count:
        raise CursorAuthorityReconciliationError(
            "global raw-frontier violation set is not exactly one cursor-ahead row"
        )
    details = _head_details(root, source_path, projection)
    plan: dict[str, object] = {
        "format": PLAN_FORMAT,
        "archive_identity": {"root": str(root.resolve())},
        "code_sha": _code_sha(),
        "deployed_package_sha": _deployed_package_sha(),
        "tier_fingerprints": tiers,
        "source_schema_versions": {tier: tiers[tier]["user_version"] for tier in _REQUIRED_TIERS},
        "selected_path_digest": path_digest,
        "observed_at_ms": int(time.time() * 1000),
        "status": "planned",
        "cursor_byte_offset": details["cursor_byte_offset"],
        "accepted_frontier": details["accepted_frontier"],
        "accepted_raw_id_digest": details["accepted_raw_id_digest"],
        "accepted_blob_hash_digest": details["accepted_blob_hash_digest"],
        "source_prefix_digest": details["source_prefix_digest"],
        "before_projection": _private_projection(projection),
    }
    plan["plan_digest"] = _canonical_digest(plan)
    return plan


def _load_plan(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CursorAuthorityReconciliationError(f"invalid reconciliation plan: {path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != PLAN_FORMAT:
        raise CursorAuthorityReconciliationError("unsupported reconciliation plan format")
    digest = payload.get("plan_digest")
    unsigned = dict(payload)
    unsigned.pop("plan_digest", None)
    if not isinstance(digest, str) or _canonical_digest(unsigned) != digest:
        raise CursorAuthorityReconciliationError("reconciliation plan digest mismatch")
    return payload


def _backup_root(manifest_path: Path) -> Path:
    root = manifest_path if manifest_path.is_dir() else manifest_path.parent
    if not root.is_dir() or not (root / "manifest.json").is_file():
        raise CursorAuthorityReconciliationError("backup manifest must be a verified full-evidence backup directory")
    return root


def _validate_backup(manifest_path: Path, plan: Mapping[str, object]) -> dict[str, object]:
    root = _backup_root(manifest_path)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        receipt = json.loads((root / "verification-receipt.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CursorAuthorityReconciliationError("backup lacks a readable verification receipt") from exc
    if not isinstance(manifest, dict) or manifest.get("profile") != "full_evidence":
        raise CursorAuthorityReconciliationError("apply requires a full_evidence backup")
    if not isinstance(receipt, dict) or receipt.get("verdict") != "success":
        raise CursorAuthorityReconciliationError("backup verification receipt is not successful")
    included = {str(item) for item in manifest.get("included_tiers", []) if isinstance(item, str)}
    if {f"{tier}.db" for tier in _REQUIRED_TIERS} - included:
        raise CursorAuthorityReconciliationError("full-evidence backup lacks source/index/ops/audit rollback evidence")
    verification = receipt.get("verification")
    required_verification = ("source_blobs_resolved", "index_attachment_blobs_resolved", "blob_inventory_exact")
    if not isinstance(verification, dict) or any(verification.get(key) is not True for key in required_verification):
        raise CursorAuthorityReconciliationError("backup lacks complete blob rollback evidence")
    if not (root / "blob").is_dir() or not (root / "blob-inventory.json").is_file():
        raise CursorAuthorityReconciliationError("backup lacks blob rollback evidence")
    declared = manifest.get("tier_source_fingerprints")
    expected = plan.get("tier_fingerprints")
    if not isinstance(declared, dict) or not isinstance(expected, dict):
        raise CursorAuthorityReconciliationError("plan or backup lacks tier fingerprints")
    for tier in _REQUIRED_TIERS:
        artifact = declared.get(f"{tier}.db")
        expected_tier = expected.get(tier)
        if not isinstance(artifact, dict) or not isinstance(expected_tier, dict):
            raise CursorAuthorityReconciliationError(f"backup lacks {tier} fingerprint")
        for key in ("size_bytes", "sha256", "user_version"):
            if artifact.get(key) != expected_tier.get(key):
                raise CursorAuthorityReconciliationError(f"backup {tier} fingerprint does not match the plan")
        backup_tier = root / f"{tier}.db"
        if not backup_tier.is_file():
            raise CursorAuthorityReconciliationError(f"backup {tier} tier is missing")
        actual_size, actual_sha256 = _file_fingerprint(backup_tier)
        if actual_sha256 != str(expected_tier["sha256"]) or actual_size != int(expected_tier["size_bytes"]):
            raise CursorAuthorityReconciliationError(f"backup {tier} bytes do not match the plan fingerprint")
    return {"root": str(root.resolve()), "manifest_sha256": _sha256_file(root / "manifest.json")}


def _quick_checks(root: Path) -> dict[str, list[str]]:
    checks: dict[str, list[str]] = {}
    for tier in ("source", "index", "ops", "audit"):
        with closing(sqlite3.connect(f"file:{(root / f'{tier}.db').resolve()}?mode=ro", uri=True)) as conn:
            checks[tier] = [str(row[0]) for row in conn.execute("PRAGMA quick_check")]
        if checks[tier] != ["ok"]:
            raise CursorAuthorityReconciliationError(f"{tier}.db quick_check failed: {checks[tier]}")
    return checks


def _write_atomic_json(path: Path, payload: Mapping[str, object], *, refuse_existing: bool) -> None:
    if refuse_existing and path.exists():
        raise CursorAuthorityReconciliationError(f"output path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if refuse_existing:
            try:
                os.link(temporary_path, path)
            except FileExistsError as exc:
                raise CursorAuthorityReconciliationError(f"output path already exists: {path}") from exc
            temporary_path.unlink()
        else:
            os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _require_daemon_stopped(root: Path) -> None:
    config = Config(archive_root=root, render_root=root / "render", sources=[], db_path=root / "index.db")
    from polylogue.maintenance.offline_guard import running_daemon_pid

    if running_daemon_pid(config) is not None:
        raise CursorAuthorityReconciliationError("daemon must be stopped for cursor-authority reconciliation")


def build_reconciliation_plan(*, source_path_file: Path, output_plan: Path) -> dict[str, object]:
    root = _archive_root()
    _require_daemon_stopped(root)
    source_path = _read_private_source_path(source_path_file)
    plan = _build_plan(root, source_path)
    _write_atomic_json(output_plan, plan, refuse_existing=True)
    return plan


def _find_recovery_attempt(root: Path, source_path: Path, plan_observed_at_ms: int) -> str | None:
    with closing(sqlite3.connect(f"file:{(root / 'ops.db').resolve()}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT attempt_id, status, source_path, source_paths_json, finished_at_ms FROM ingest_attempts "
            "ORDER BY COALESCE(finished_at_ms, heartbeat_at_ms, started_at_ms) DESC LIMIT 50"
        ).fetchall()
    for attempt_id, status, single_path, paths_json, finished_at_ms in rows:
        if str(status) not in {"completed", "completed_with_failures"}:
            continue
        if not isinstance(finished_at_ms, int) or finished_at_ms <= plan_observed_at_ms:
            continue
        values: list[str] = []
        if isinstance(paths_json, str):
            try:
                decoded = json.loads(paths_json)
            except ValueError:
                decoded = []
            if isinstance(decoded, list):
                values.extend(str(value) for value in decoded if isinstance(value, str))
        if not values and single_path:
            values.append(str(single_path))
        if any(Path(value).resolve() == source_path.resolve() for value in values):
            return str(attempt_id)
    return None


async def _normal_ingest(root: Path, source_path: Path, plan: Mapping[str, object]) -> tuple[LiveBatchMetrics, str]:
    from polylogue.sources.live import watcher as live_watcher

    async with Polylogue(archive_root=root, db_path=root / "index.db") as polylogue:
        cursor = CursorStore(root / "ops.db", initialize=False, ops_db_path=root / "ops.db")
        processor = LiveBatchProcessor(
            polylogue,
            (WatchSource(name=source_path.parent.name, root=source_path.parent),),
            cursor=cursor,
            parser_fingerprint=lambda: live_watcher._PARSER_FINGERPRINT,
        )
        with scoped_cursor_authority_authorization(
            source_path_digest=str(plan["selected_path_digest"]),
            cursor_byte_offset=_plan_int(plan, "cursor_byte_offset"),
            accepted_frontier=_plan_int(plan, "accepted_frontier"),
            plan_digest=str(plan["plan_digest"]),
        ):
            metrics = await processor.ingest_files([source_path], emit_event=False)
        return metrics, _find_recovery_attempt(root, source_path, _plan_int(plan, "observed_at_ms")) or "unknown"


def _plan_int(plan: Mapping[str, object], key: str) -> int:
    value = plan.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CursorAuthorityReconciliationError(f"reconciliation plan field {key} is not an integer")
    return value


def _required_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CursorAuthorityReconciliationError(f"{field} is missing or invalid")
    return value


def _before_projection(plan: Mapping[str, object]) -> dict[str, object]:
    value = plan.get("before_projection")
    if not isinstance(value, dict):
        raise CursorAuthorityReconciliationError("plan lacks the before-projection authority census")
    return value


def _same_plan_bindings(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    def comparable(plan: Mapping[str, object]) -> dict[str, object]:
        value = dict(plan)
        value.pop("observed_at_ms", None)
        value.pop("plan_digest", None)
        return value

    return comparable(left) == comparable(right)


def _changed_rows() -> dict[str, int | None]:
    return {"cursor": None, "accepted_head_direct_writes": 0}


def _receipt_payload(
    *,
    plan: Mapping[str, object],
    backup: Mapping[str, object],
    root: Path,
    verdict: str,
    before_projection: Mapping[str, object],
    after_projection: RawFrontierIntegrityProjection | None,
    metrics: LiveBatchMetrics | None,
    attempt_id: str | None,
    attempt_observation: str,
    evidence: Mapping[str, object],
    tolerate_state_errors: bool = False,
) -> dict[str, object]:
    try:
        tier_fingerprints: object = _tier_snapshots(root)
        quick_check: object = _quick_checks(root)
    except Exception:
        if not tolerate_state_errors:
            raise
        tier_fingerprints = None
        quick_check = None
    return {
        "format": RECEIPT_FORMAT,
        "verdict": verdict,
        "archive_identity": {"root": str(root.resolve())},
        "plan_digest": plan["plan_digest"],
        "backup": dict(backup),
        "before_projection": dict(before_projection),
        "after_projection": _private_projection(after_projection) if after_projection is not None else None,
        "metrics": metrics.to_payload() if metrics is not None else None,
        "changed_rows": _changed_rows(),
        "ingest_attempt_id": attempt_id,
        "ingest_attempt_observation": attempt_observation,
        "operation": attempt_observation,
        "code_sha": plan.get("code_sha"),
        "deployed_package_sha": plan.get("deployed_package_sha"),
        "tier_fingerprints": tier_fingerprints,
        "quick_check": quick_check,
        "evidence": dict(evidence),
    }


def apply_reconciliation(*, plan_path: Path, backup_manifest: Path, receipt: Path) -> dict[str, object]:
    plan = _load_plan(plan_path)
    if plan.get("status") != "planned":
        raise CursorAuthorityReconciliationError("only a planned one-path reconciliation can be applied")
    root = _archive_root()
    _require_daemon_stopped(root)
    if receipt.exists():
        raise CursorAuthorityReconciliationError(f"output path already exists: {receipt}")
    before_projection = _before_projection(plan)
    backup_evidence = _validate_backup(backup_manifest, plan)
    current_path = _find_path_by_digest(root, str(plan["selected_path_digest"]))
    try:
        current_plan = _build_plan(root, current_path)
    except CursorAuthorityReconciliationError:
        current_plan = None
    if current_plan is None or not _same_plan_bindings(current_plan, plan):
        recovery_projection = _projection_for(root)
        recovery_attempt = _find_recovery_attempt(root, current_path, _plan_int(plan, "observed_at_ms"))
        if recovery_attempt is None or recovery_projection.cursor_ahead_count != 0:
            raise CursorAuthorityReconciliationError("plan bindings changed before archive ownership")
        before_gap_count = before_projection.get("cursor_authority_gap_count")
        if not isinstance(before_gap_count, int) or recovery_projection.cursor_authority_gap_count != before_gap_count:
            raise CursorAuthorityReconciliationError(
                "recovered ingest changed the pre-existing incomparable cursor population"
            )
        recovered_receipt_payload = _receipt_payload(
            plan=plan,
            backup=backup_evidence,
            root=root,
            verdict="reconciled" if recovery_projection.overall_status == "healthy" else "typed_deferred",
            before_projection=before_projection,
            after_projection=recovery_projection,
            metrics=None,
            attempt_id=recovery_attempt,
            attempt_observation="observed",
            evidence={
                "raw_frontier_worsening": False,
                "invalid_ahead_reconciliation": False,
                "changed_pre_existing_populations": False,
            },
        )
        recovered_receipt_payload["receipt_digest"] = _canonical_digest(recovered_receipt_payload)
        _write_atomic_json(receipt, recovered_receipt_payload, refuse_existing=True)
        return recovered_receipt_payload
    owner = acquire_durable_archive_ownership(root, owner_id=f"cursor-authority-reconcile:{os.getpid()}")
    with owner:
        current_plan = _build_plan(root, current_path)
        if not _same_plan_bindings(current_plan, plan):
            raise CursorAuthorityReconciliationError("plan bindings changed after archive ownership")
        metrics: LiveBatchMetrics | None = None
        attempt_id = "unknown"
        after_projection: RawFrontierIntegrityProjection | None = None
        evidence: dict[str, object] = {
            "raw_frontier_worsening": False,
            "invalid_ahead_reconciliation": False,
            "changed_pre_existing_populations": False,
        }
        try:
            metrics, attempt_id = asyncio.run(_normal_ingest(root, current_path, plan))
            after_projection = _projection_for(root)
            if after_projection.broken_head_count or after_projection.missing_source_raw_count:
                evidence["raw_frontier_worsening"] = True
                raise CursorAuthorityReconciliationError("reconciliation introduced unrelated raw-frontier worsening")
            if after_projection.cursor_ahead_count:
                if (
                    metrics.succeeded_file_count != 0
                    or str(current_path) not in metrics.failed_paths
                    or metrics.time_budget_exceeded
                ):
                    evidence["invalid_ahead_reconciliation"] = True
                    raise CursorAuthorityReconciliationError(
                        "cursor-ahead postcondition is neither reconciled nor explicitly deferred"
                    )
                verdict = "typed_deferred"
            else:
                verdict = "reconciled" if after_projection.overall_status == "healthy" else "typed_deferred"
            before_gap_count = before_projection.get("cursor_authority_gap_count")
            if not isinstance(before_gap_count, int) or after_projection.cursor_authority_gap_count != before_gap_count:
                evidence["changed_pre_existing_populations"] = True
                raise CursorAuthorityReconciliationError(
                    "reconciliation changed the pre-existing incomparable cursor population"
                )
            receipt_payload = _receipt_payload(
                plan=plan,
                backup=backup_evidence,
                root=root,
                verdict=verdict,
                before_projection=before_projection,
                after_projection=after_projection,
                metrics=metrics,
                attempt_id=attempt_id,
                attempt_observation="performed",
                evidence=evidence,
                tolerate_state_errors=True,
            )
        except Exception as exc:
            if after_projection is None:
                try:
                    after_projection = _projection_for(root)
                except Exception:
                    after_projection = None
            failure_payload = _receipt_payload(
                plan=plan,
                backup=backup_evidence,
                root=root,
                verdict="failed",
                before_projection=before_projection,
                after_projection=after_projection,
                metrics=metrics,
                attempt_id=attempt_id,
                attempt_observation="performed",
                evidence=evidence,
            )
            failure_payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
            failure_payload["receipt_digest"] = _canonical_digest(failure_payload)
            _write_atomic_json(receipt, failure_payload, refuse_existing=True)
            if isinstance(exc, CursorAuthorityReconciliationError):
                raise
            raise CursorAuthorityReconciliationError("cursor-authority reconciliation failed after ingest") from exc
        receipt_payload["receipt_digest"] = _canonical_digest(receipt_payload)
        _write_atomic_json(receipt, receipt_payload, refuse_existing=True)
        return receipt_payload


__all__ = [
    "ARCHIVE_ROOT",
    "PLAN_FORMAT",
    "RECEIPT_FORMAT",
    "CursorAuthorityReconciliationError",
    "apply_reconciliation",
    "build_reconciliation_plan",
    "cursor_authority_path_digest",
]

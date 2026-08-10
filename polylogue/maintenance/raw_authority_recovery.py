"""Guarded, receipted recovery for the raw-authority derived state.

This is the only production mutation route for the two break-glass repairs in
this module.  The storage helpers expose counts for diagnostics, but they do
not authorize deletion.  A plan is an exact snapshot of one archive and one
operation.  APPLY rechecks that snapshot while holding both archive ownership
and the rebuild lease, then performs one transaction against one tier.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat as stat_module
import tempfile
import uuid
from collections.abc import Generator, Mapping
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from polylogue.config import Config
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason, running_daemon_pid
from polylogue.operations.mutation_transaction import (
    ConfirmationStrength,
    DestructiveClass,
    MutationPlan,
    MutationReceipt,
    MutationTransactionError,
    OperationExecutor,
    PlanStaleError,
    build_plan,
    make_target_ref,
)
from polylogue.paths import render_root
from polylogue.storage.archive_identity import (
    ArchiveLocation,
    ArchiveLocationError,
    ArchiveOwnershipError,
    OwnedArchiveLocation,
    assert_owns_archive_location,
)
from polylogue.storage.index_generation import RebuildLease, source_revision_snapshot
from polylogue.storage.introspection import table_exists
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import (
    validate_backup_manifest_covers_derived_tier,
    validate_migration_backup_manifest,
)
from polylogue.version import VERSION_INFO


class RecoveryOperation(StrEnum):
    RESET_CENSUS = "reset_raw_authority_census"
    PRUNE_INDEX_SEEDS = "prune_orphaned_index_revision_seeds"


PLAN_FORMAT = "polylogue.raw-authority-recovery-plan.v1"
RECEIPT_FORMAT = "polylogue.raw-authority-recovery-receipt.v1"
INTENT_FORMAT = "polylogue.raw-authority-recovery-intent.v1"
RECOVERY_DIRNAME = "raw-authority-recovery"
_RESET_TABLES = (
    "raw_authority_blockers",
    "raw_authority_census_plans",
    "raw_authority_census_post_plans",
    "raw_authority_plans",
    "raw_authority_censuses",
)
_INDEX_TARGETS = ("raw_revision_heads", "raw_revision_applications")


class RawAuthorityRecoveryError(RuntimeError):
    """A recovery plan or apply could not prove its safety contract."""


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_fingerprint(path: Path) -> dict[str, object]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise RawAuthorityRecoveryError(f"recovery tier is not readable: {path}") from exc
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RawAuthorityRecoveryError(f"could not fingerprint recovery tier: {path}") from exc
    return {
        "path": str(path.resolve(strict=False)),
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _pointer_fingerprint(root: Path) -> dict[str, object]:
    pointer = root / ".index-active-pointer"
    if not pointer.exists() and not pointer.is_symlink():
        return {"path": str(pointer), "exists": False, "text": None}
    try:
        text = pointer.read_text(encoding="utf-8")
    except OSError as exc:
        raise RawAuthorityRecoveryError(f"active index pointer is unreadable: {pointer}") from exc
    return {"path": str(pointer), "exists": True, "text": text}


def _quote_identifier(name: str) -> str:
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for char in name):
        raise RawAuthorityRecoveryError(f"unexpected SQLite identifier: {name!r}")
    return f'"{name}"'


def _ensure_tables(conn: sqlite3.Connection, names: tuple[str, ...], *, tier: str) -> None:
    missing = [name for name in names if not table_exists(conn, name)]
    if missing:
        raise RawAuthorityRecoveryError(f"{tier} database is missing required table(s): {', '.join(missing)}")


def _value_for_digest(value: object) -> object:
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__, "value": str(value)}


def _table_digest(conn: sqlite3.Connection, name: str) -> str:
    quoted = _quote_identifier(name)
    columns = [str(row[1]) for row in conn.execute(f"PRAGMA table_info({quoted})")]
    if not columns:
        raise RawAuthorityRecoveryError(f"cannot fingerprint missing or malformed table: {name}")
    rows = [[_value_for_digest(value) for value in row] for row in conn.execute(f"SELECT * FROM {quoted}")]
    rows.sort(key=_canonical_bytes)
    return _digest({"name": name, "columns": columns, "rows": rows})


def _protected_digest(conn: sqlite3.Connection, *, excluded: tuple[str, ...]) -> str:
    tables = sorted(
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
        if str(row[0]) not in excluded
    )
    return _digest({name: _table_digest(conn, name) for name in tables})


def _schema_versions(root: Path, location: ArchiveLocation) -> dict[str, int]:
    paths = {
        ArchiveTier.SOURCE.value: root / "source.db",
        ArchiveTier.INDEX.value: location.active_index_path,
        ArchiveTier.EMBEDDINGS.value: root / "embeddings.db",
        ArchiveTier.USER.value: root / "user.db",
        ArchiveTier.OPS.value: root / "ops.db",
        ArchiveTier.AUDIT.value: root / "audit.db",
    }
    versions: dict[str, int] = {}
    for tier, path in paths.items():
        if not path.is_file():
            raise RawAuthorityRecoveryError(f"archive tier is missing: {path}")
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            versions[tier] = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
    expected = {tier.value: version for tier, version in ARCHIVE_VERSION_BY_TIER.items()}
    if versions != expected:
        raise RawAuthorityRecoveryError(
            f"archive schema versions are not current: observed={versions}, expected={expected}"
        )
    return versions


def _archive_identity(root: Path, location: ArchiveLocation) -> dict[str, object]:
    from polylogue.storage.archive_identity import ArchiveIdentity

    identity = ArchiveIdentity.resolve_location(location)
    return identity.as_dict(unit="raw-authority-recovery")


def _generation_identity(root: Path, location: ArchiveLocation) -> dict[str, object]:
    payload: dict[str, object] = {
        "active_generation": location.active_generation,
        "active_index_path": str(location.active_index_path.resolve(strict=False)),
        "active_index_stable_id": location.active_index.stable_id,
        "pointer": _pointer_fingerprint(root),
    }
    metadata = location.active_index_path.parent / "generation.json"
    if metadata.is_file():
        try:
            decoded = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RawAuthorityRecoveryError(f"active generation metadata is malformed: {metadata}") from exc
        if not isinstance(decoded, dict):
            raise RawAuthorityRecoveryError(f"active generation metadata is not an object: {metadata}")
        payload["metadata"] = {str(key): value for key, value in decoded.items()}
    return payload


def _validate_ledger(conn: sqlite3.Connection) -> None:
    _ensure_tables(conn, ("raw_authority_parser_census", *_RESET_TABLES, "raw_sessions"), tier="source")
    for table, columns in {
        "raw_authority_censuses": ("scope_json", "residual_json"),
        "raw_authority_plans": (
            "input_raw_ids_json",
            "logical_keys_json",
            "authority_witness_json",
            "source_preconditions_json",
            "index_preconditions_json",
        ),
        "raw_authority_census_plans": ("application_receipt_json",),
        "raw_authority_blockers": ("expected_json", "observed_json"),
    }.items():
        quoted = _quote_identifier(table)
        for row in conn.execute(f"SELECT * FROM {quoted}"):
            names = [str(item[1]) for item in conn.execute(f"PRAGMA table_info({quoted})")]
            values = dict(zip(names, row, strict=True))
            for column in columns:
                try:
                    decoded = json.loads(str(values[column]))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise RawAuthorityRecoveryError(f"raw-authority ledger has malformed {table}.{column}") from exc
                if not isinstance(decoded, (dict, list)):
                    raise RawAuthorityRecoveryError(f"raw-authority ledger has unexpected {table}.{column} JSON class")
    if list(conn.execute("PRAGMA foreign_key_check")):
        raise RawAuthorityRecoveryError("raw-authority ledger has foreign-key violations")


def _validate_integrity(conn: sqlite3.Connection, *, tier: str) -> None:
    quick_check = tuple(str(row[0]) for row in conn.execute("PRAGMA quick_check"))
    foreign_keys = tuple(tuple(str(value) for value in row) for row in conn.execute("PRAGMA foreign_key_check"))
    if quick_check != ("ok",):
        raise RawAuthorityRecoveryError(f"{tier} database quick_check failed: {quick_check}")
    if foreign_keys:
        raise RawAuthorityRecoveryError(f"{tier} database has foreign-key violations: {foreign_keys[:3]}")


def _count_tables(conn: sqlite3.Connection, names: tuple[str, ...]) -> dict[str, int]:
    _ensure_tables(conn, names, tier="archive")
    return {name: int(conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(name)}").fetchone()[0]) for name in names}


def _index_candidates(conn: sqlite3.Connection) -> dict[str, tuple[str, ...]]:
    _ensure_tables(conn, _INDEX_TARGETS, tier="index")
    heads = tuple(
        str(row[0])
        for row in conn.execute(
            "SELECT h.logical_source_key FROM raw_revision_heads AS h "
            "WHERE NOT EXISTS (SELECT 1 FROM src.raw_sessions AS r WHERE r.raw_id = h.accepted_raw_id) "
            "ORDER BY h.logical_source_key"
        )
    )
    applications = tuple(
        str(row[0])
        for row in conn.execute(
            "SELECT a.decision_id FROM raw_revision_applications AS a "
            "WHERE NOT EXISTS (SELECT 1 FROM src.raw_sessions AS r WHERE r.raw_id = a.raw_id) "
            "ORDER BY a.decision_id"
        )
    )
    if any(not value for value in (*heads, *applications)):
        raise RawAuthorityRecoveryError("orphaned index revision seed has an empty primary key")
    return {"raw_revision_heads": heads, "raw_revision_applications": applications}


def _index_seed_digest(conn: sqlite3.Connection, *, excluded_keys: Mapping[str, tuple[str, ...]] | None = None) -> str:
    """Hash the exact index-seed rows expected after a guarded prune."""

    key_columns = {
        "raw_revision_heads": "logical_source_key",
        "raw_revision_applications": "decision_id",
    }
    payload: dict[str, object] = {}
    for table, key_column in key_columns.items():
        quoted = _quote_identifier(table)
        columns = [str(row[1]) for row in conn.execute(f"PRAGMA table_info({quoted})")]
        if not columns:
            raise RawAuthorityRecoveryError(f"cannot fingerprint missing or malformed table: {table}")
        excluded = () if excluded_keys is None else excluded_keys.get(table, ())
        if excluded:
            placeholders = ", ".join("?" for _ in excluded)
            rows_query = f"SELECT * FROM {quoted} WHERE {_quote_identifier(key_column)} NOT IN ({placeholders})"
            rows = conn.execute(rows_query, excluded)
        else:
            rows = conn.execute(f"SELECT * FROM {quoted}")
        values = [[_value_for_digest(value) for value in row] for row in rows]
        values.sort(key=_canonical_bytes)
        payload[table] = {"columns": columns, "rows": values}
    return _digest(payload)


def _validate_backup(path: Path | None, *, tier: ArchiveTier, connection: sqlite3.Connection) -> dict[str, object]:
    if path is None:
        raise RawAuthorityRecoveryError(f"{tier.value}-tier backup authority is required for apply")
    manifest = path.expanduser().resolve(strict=False)
    try:
        receipt = (
            validate_migration_backup_manifest(manifest, tier, connection=connection)
            if tier in {ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT}
            else validate_backup_manifest_covers_derived_tier(manifest, tier, connection=connection)
        )
    except Exception as exc:
        raise RawAuthorityRecoveryError(f"backup authority refused for {tier.value}: {exc}") from exc
    receipt = receipt.resolve(strict=False)
    return {
        "tier": tier.value,
        "manifest_path": str(manifest),
        "manifest_sha256": _file_fingerprint(manifest)["sha256"],
        "receipt_path": str(receipt),
        "receipt_sha256": _file_fingerprint(receipt)["sha256"],
    }


def _backup_from_plan(plan: RawAuthorityRecoveryPlan, *, connection: sqlite3.Connection) -> None:
    authority = plan.backup_authority
    if not isinstance(authority, dict):
        raise RawAuthorityRecoveryError("apply plan has no verified backup authority")
    manifest_path = Path(str(authority.get("manifest_path", "")))
    if not manifest_path.is_file() or _file_fingerprint(manifest_path)["sha256"] != authority.get("manifest_sha256"):
        raise RawAuthorityRecoveryError("authorized backup manifest changed or is missing")
    receipt_path = Path(str(authority.get("receipt_path", "")))
    if not receipt_path.is_file() or _file_fingerprint(receipt_path)["sha256"] != authority.get("receipt_sha256"):
        raise RawAuthorityRecoveryError("authorized backup verification receipt changed or is missing")
    tier = ArchiveTier(str(authority.get("tier", "")))
    refreshed = _validate_backup(manifest_path, tier=tier, connection=connection)
    if refreshed != authority:
        raise RawAuthorityRecoveryError("backup authority no longer matches the recovery plan")


def _postflight(conn: sqlite3.Connection, *, protected_digest: str, excluded: tuple[str, ...]) -> dict[str, object]:
    quick_check = tuple(str(row[0]) for row in conn.execute("PRAGMA quick_check"))
    foreign_keys = tuple(tuple(str(value) for value in row) for row in conn.execute("PRAGMA foreign_key_check"))
    actual_protected = _protected_digest(conn, excluded=excluded)
    if quick_check != ("ok",):
        raise RawAuthorityRecoveryError(f"postflight quick_check failed: {quick_check}")
    if foreign_keys:
        raise RawAuthorityRecoveryError(f"postflight foreign_key_check failed: {foreign_keys[:3]}")
    if actual_protected != protected_digest:
        raise RawAuthorityRecoveryError("postflight changed an unrelated table")
    return {
        "quick_check": list(quick_check),
        "foreign_key_check": [list(row) for row in foreign_keys],
        "protected_digest": actual_protected,
    }


@dataclass(frozen=True, slots=True)
class RawAuthorityRecoveryPlan:
    operation_id: str
    operation: str
    archive_root: str
    archive_identity: dict[str, object]
    archive_identity_digest: str
    schema_versions: dict[str, int]
    code_sha: str
    source_fingerprint: dict[str, object]
    index_fingerprint: dict[str, object]
    source_snapshot: str
    active_generation: dict[str, object]
    before_counts: dict[str, int]
    candidate_keys: dict[str, tuple[str, ...]]
    post_target_digest: str | None
    protected_digest: str
    backup_authority: dict[str, object] | None
    receipt_path: str
    plan_digest: str

    def _payload(self) -> dict[str, object]:
        return {
            "format": PLAN_FORMAT,
            "operation_id": self.operation_id,
            "operation": self.operation,
            "archive_root": self.archive_root,
            "archive_identity": self.archive_identity,
            "archive_identity_digest": self.archive_identity_digest,
            "schema_versions": self.schema_versions,
            "code_sha": self.code_sha,
            "source_fingerprint": self.source_fingerprint,
            "index_fingerprint": self.index_fingerprint,
            "source_snapshot": self.source_snapshot,
            "active_generation": self.active_generation,
            "before_counts": self.before_counts,
            "candidate_keys": {key: list(value) for key, value in self.candidate_keys.items()},
            "post_target_digest": self.post_target_digest,
            "protected_digest": self.protected_digest,
            "backup_authority": self.backup_authority,
            "receipt_path": self.receipt_path,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "plan_digest": self.plan_digest}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RawAuthorityRecoveryPlan:
        if payload.get("format") != PLAN_FORMAT:
            raise RawAuthorityRecoveryError("unsupported or missing raw-authority recovery plan format")
        expected = payload.get("plan_digest")
        actual = _digest({key: value for key, value in payload.items() if key != "plan_digest"})
        if not isinstance(expected, str) or expected != actual:
            raise RawAuthorityRecoveryError("raw-authority recovery plan digest is invalid")
        candidates_raw = payload.get("candidate_keys")
        if not isinstance(candidates_raw, dict):
            raise RawAuthorityRecoveryError("raw-authority recovery plan candidate keys are malformed")
        if any(not isinstance(value_list, list) for value_list in candidates_raw.values()):
            raise RawAuthorityRecoveryError("raw-authority recovery plan candidate keys are malformed")
        candidates = {
            str(key): tuple(str(value) for value in cast(list[object], value_list))
            for key, value_list in candidates_raw.items()
        }
        required = (
            "operation_id",
            "operation",
            "archive_root",
            "archive_identity",
            "archive_identity_digest",
            "schema_versions",
            "code_sha",
            "source_fingerprint",
            "index_fingerprint",
            "source_snapshot",
            "active_generation",
            "before_counts",
            "post_target_digest",
            "protected_digest",
            "receipt_path",
        )
        if any(key not in payload for key in required):
            raise RawAuthorityRecoveryError("raw-authority recovery plan is missing a required field")
        post_target_digest = payload["post_target_digest"]
        if post_target_digest is not None and not isinstance(post_target_digest, str):
            raise RawAuthorityRecoveryError("raw-authority recovery plan post-target digest is malformed")
        return cls(
            operation_id=str(payload["operation_id"]),
            operation=str(payload["operation"]),
            archive_root=str(payload["archive_root"]),
            archive_identity=cast(dict[str, object], payload["archive_identity"]),
            archive_identity_digest=str(payload["archive_identity_digest"]),
            schema_versions=_int_mapping(payload["schema_versions"], field="schema_versions"),
            code_sha=str(payload["code_sha"]),
            source_fingerprint=cast(dict[str, object], payload["source_fingerprint"]),
            index_fingerprint=cast(dict[str, object], payload["index_fingerprint"]),
            source_snapshot=str(payload["source_snapshot"]),
            active_generation=cast(dict[str, object], payload["active_generation"]),
            before_counts=_int_mapping(payload["before_counts"], field="before_counts"),
            candidate_keys=candidates,
            post_target_digest=post_target_digest,
            protected_digest=str(payload["protected_digest"]),
            backup_authority=(
                cast(dict[str, object], payload["backup_authority"])
                if isinstance(payload.get("backup_authority"), dict)
                else None
            ),
            receipt_path=str(payload["receipt_path"]),
            plan_digest=actual,
        )


@dataclass(frozen=True, slots=True)
class RawAuthorityRecoveryReport:
    plan: RawAuthorityRecoveryPlan
    applied: bool
    status: str
    receipt_path: Path | None = None
    after_counts: dict[str, int] | None = None
    postflight: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "operation_id": self.plan.operation_id,
            "operation": self.plan.operation,
            "plan": self.plan.to_dict(),
            "applied": self.applied,
            "status": self.status,
            "receipt_path": str(self.receipt_path) if self.receipt_path is not None else None,
            "after_counts": self.after_counts,
            "postflight": self.postflight,
        }


def _default_receipt_path(root: Path, operation_id: str) -> Path:
    return root / ".maintenance-state" / RECOVERY_DIRNAME / f"{operation_id}.receipt.json"


def _resolve_receipt_path(root: Path, receipt_path: Path) -> Path:
    """Keep recovery evidence inside the archive's durable maintenance state."""

    archive_root = root.expanduser().resolve(strict=False)
    receipts_root = archive_root / ".maintenance-state" / RECOVERY_DIRNAME
    candidate = Path(os.path.abspath(receipt_path.expanduser()))
    try:
        candidate.relative_to(receipts_root)
    except ValueError as exc:
        raise RawAuthorityRecoveryError(
            f"recovery receipt path must be inside the archive-owned durable location {receipts_root}"
        ) from exc
    if candidate.suffix != ".json":
        raise RawAuthorityRecoveryError("recovery receipt path must end in .json")
    current = archive_root
    for component in candidate.relative_to(archive_root).parts:
        current /= component
        if current.is_symlink():
            raise RawAuthorityRecoveryError(f"recovery receipt path must not traverse an archive symlink: {current}")
    return candidate


def _intent_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(f"{receipt_path.name}.intent.json")


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RawAuthorityRecoveryError(f"recovery plan or receipt is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RawAuthorityRecoveryError(f"recovery artifact is not a JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


@contextmanager
def _durable_receipt_directory(
    root: Path, receipt_path: Path, *, create: bool
) -> Generator[tuple[int, str], None, None]:
    """Open a receipt parent by descriptor without following archive symlinks."""

    candidate = _resolve_receipt_path(root, receipt_path)
    archive_root = root.expanduser().resolve(strict=False)
    current_fd = os.open(archive_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in candidate.parent.relative_to(archive_root).parts:
            try:
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                created = False
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                    created = True
                except FileExistsError:
                    pass
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current_fd)
                if created:
                    os.fsync(current_fd)
                    os.fsync(next_fd)
            os.close(current_fd)
            current_fd = next_fd
        yield current_fd, candidate.name
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise RawAuthorityRecoveryError(
            f"recovery receipt path must remain in the archive-owned durable location: {candidate}"
        ) from exc
    finally:
        os.close(current_fd)


def _read_json_at(directory_fd: int, name: str, *, display_path: Path) -> dict[str, object]:
    try:
        artifact_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise RawAuthorityRecoveryError(
            f"recovery artifact is not a regular archive-owned file: {display_path}"
        ) from exc
    try:
        if not stat_module.S_ISREG(os.fstat(artifact_fd).st_mode):
            raise RawAuthorityRecoveryError(f"recovery artifact is not a regular archive-owned file: {display_path}")
        with os.fdopen(artifact_fd, "r", encoding="utf-8") as handle:
            artifact_fd = -1
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RawAuthorityRecoveryError(f"recovery plan or receipt is unreadable: {display_path}") from exc
    finally:
        if artifact_fd >= 0:
            os.close(artifact_fd)
    if not isinstance(payload, dict):
        raise RawAuthorityRecoveryError(f"recovery artifact is not a JSON object: {display_path}")
    return {str(key): value for key, value in payload.items()}


def _read_durable_json(root: Path, path: Path) -> dict[str, object] | None:
    try:
        with _durable_receipt_directory(root, path, create=False) as (directory_fd, name):
            try:
                return _read_json_at(directory_fd, name, display_path=path)
            except FileNotFoundError:
                return None
    except FileNotFoundError:
        return None


def _int_mapping(payload: object, *, field: str) -> dict[str, int]:
    if not isinstance(payload, dict) or any(not isinstance(value, int) for value in payload.values()):
        raise RawAuthorityRecoveryError(f"raw-authority recovery plan field {field!r} is malformed")
    return {str(key): int(value) for key, value in payload.items()}


def _write_immutable(path: Path, payload: dict[str, object], *, digest_field: str) -> Path:
    body = {key: value for key, value in payload.items() if key != digest_field}
    expected = _digest(body)
    stamped = {**body, digest_field: expected}
    path = path.expanduser().resolve(strict=False)
    if path.exists():
        existing = _read_json(path)
        if existing != stamped:
            raise RawAuthorityRecoveryError(
                f"immutable recovery artifact already exists with different content: {path}"
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)[1])
    try:
        temporary.write_text(json.dumps(stamped, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError:
        existing = _read_json(path)
        if existing != stamped:
            raise RawAuthorityRecoveryError(f"immutable recovery artifact race changed content: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_durable_immutable(root: Path, path: Path, payload: dict[str, object], *, digest_field: str) -> Path:
    """Publish a self-hashed receipt without pathname traversal after validation."""

    body = {key: value for key, value in payload.items() if key != digest_field}
    stamped = {**body, digest_field: _digest(body)}
    serialized = (json.dumps(stamped, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with _durable_receipt_directory(root, path, create=True) as (directory_fd, name):
        try:
            existing = _read_json_at(directory_fd, name, display_path=path)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != stamped:
                raise RawAuthorityRecoveryError(
                    f"immutable recovery artifact already exists with different content: {path}"
                )
            return path

        temporary_name = f".{name}.{uuid.uuid4().hex}.tmp"
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            view = memoryview(serialized)
            while view:
                view = view[os.write(temporary_fd, view) :]
            os.fsync(temporary_fd)
            os.link(
                temporary_name,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            os.fsync(directory_fd)
        except FileExistsError:
            existing = _read_json_at(directory_fd, name, display_path=path)
            if existing != stamped:
                raise RawAuthorityRecoveryError(f"immutable recovery artifact race changed content: {path}") from None
        finally:
            os.close(temporary_fd)
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_fd)
    return path


def write_recovery_plan(plan: RawAuthorityRecoveryPlan, path: Path) -> Path:
    return _write_immutable(path, plan.to_dict(), digest_field="plan_digest")


def _load_plan(path: Path) -> RawAuthorityRecoveryPlan:
    return RawAuthorityRecoveryPlan.from_dict(_read_json(path))


def _build_plan(
    archive_root: Path,
    *,
    operation: RecoveryOperation,
    operation_id: str,
    backup_manifest: Path | None,
    receipt_path: Path | None,
) -> RawAuthorityRecoveryPlan:
    root = archive_root.expanduser().resolve(strict=False)
    location = ArchiveLocation.resolve(root)
    source_db = root / "source.db"
    index_db = location.active_index_path
    if not source_db.is_file() or not index_db.is_file():
        raise FileNotFoundError(source_db if not source_db.is_file() else index_db)
    schema_versions = _schema_versions(root, location)
    code_sha = VERSION_INFO.commit
    if not code_sha:
        raise RawAuthorityRecoveryError("recovery requires an exact build code SHA")
    identity = _archive_identity(root, location)
    identity_digest = _digest(identity)
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as source:
        source.row_factory = sqlite3.Row
        source_counts = _count_tables(source, _RESET_TABLES)
        if operation is RecoveryOperation.RESET_CENSUS:
            _validate_ledger(source)
            _validate_integrity(source, tier="source")
            counts = source_counts
            candidate_keys: dict[str, tuple[str, ...]] = {}
            post_target_digest: str | None = None
            protected = _protected_digest(source, excluded=_RESET_TABLES)
        else:
            with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as index:
                index.row_factory = sqlite3.Row
                index.execute("ATTACH DATABASE ? AS src", (str(source_db),))
                _validate_integrity(index, tier="active index")
                candidate_keys = _index_candidates(index)
                counts = _count_tables(index, _INDEX_TARGETS)
                post_target_digest = _index_seed_digest(index, excluded_keys=candidate_keys)
                protected = _protected_digest(index, excluded=_INDEX_TARGETS)
        source_snapshot = source_revision_snapshot(root)
    backup_authority: dict[str, object] | None = None
    if backup_manifest is not None:
        target = ArchiveTier.SOURCE if operation is RecoveryOperation.RESET_CENSUS else ArchiveTier.INDEX
        with closing(
            sqlite3.connect(f"file:{source_db if target is ArchiveTier.SOURCE else index_db}?mode=ro", uri=True)
        ) as conn:
            backup_authority = _validate_backup(backup_manifest, tier=target, connection=conn)
    receipt = _resolve_receipt_path(root, receipt_path or _default_receipt_path(root, operation_id))
    payload: dict[str, object] = {
        "format": PLAN_FORMAT,
        "operation_id": operation_id,
        "operation": operation.value,
        "archive_root": str(root),
        "archive_identity": identity,
        "archive_identity_digest": identity_digest,
        "schema_versions": schema_versions,
        "code_sha": code_sha,
        "source_fingerprint": _file_fingerprint(source_db),
        "index_fingerprint": _file_fingerprint(index_db),
        "source_snapshot": source_snapshot,
        "active_generation": _generation_identity(root, location),
        "before_counts": counts,
        "candidate_keys": {key: list(value) for key, value in candidate_keys.items()},
        "post_target_digest": post_target_digest,
        "protected_digest": protected,
        "backup_authority": backup_authority,
        "receipt_path": str(receipt),
    }
    return RawAuthorityRecoveryPlan(
        operation_id=operation_id,
        operation=operation.value,
        archive_root=str(root),
        archive_identity=identity,
        archive_identity_digest=identity_digest,
        schema_versions=schema_versions,
        code_sha=code_sha,
        source_fingerprint=cast(dict[str, object], payload["source_fingerprint"]),
        index_fingerprint=cast(dict[str, object], payload["index_fingerprint"]),
        source_snapshot=source_snapshot,
        active_generation=cast(dict[str, object], payload["active_generation"]),
        before_counts=counts,
        candidate_keys=candidate_keys,
        post_target_digest=post_target_digest,
        protected_digest=protected,
        backup_authority=backup_authority,
        receipt_path=str(receipt),
        plan_digest=_digest(payload),
    )


def inspect_raw_authority_recovery(
    archive_root: Path,
    operation: RecoveryOperation | str,
    *,
    operation_id: str | None = None,
    backup_manifest: Path | None = None,
    receipt_path: Path | None = None,
) -> RawAuthorityRecoveryPlan:
    """Build a read-only, exact plan for one recovery operation."""
    selected = RecoveryOperation(operation)
    return _build_plan(
        archive_root,
        operation=selected,
        operation_id=operation_id or f"raw-authority-recovery:{uuid.uuid4().hex}",
        backup_manifest=backup_manifest,
        receipt_path=receipt_path,
    )


def _offline_config(root: Path) -> Config:
    return Config(archive_root=root, render_root=render_root(), sources=[])


def _require_apply_preconditions(root: Path) -> None:
    if running_daemon_pid(_offline_config(root)) is not None:
        raise RawAuthorityRecoveryError("refusing raw-authority recovery while polylogued is running")
    if reason := offline_maintenance_block_reason(_offline_config(root), active=True, dry_run=False):
        raise RawAuthorityRecoveryError(reason)


def _same(value: object, expected: object, field: str) -> None:
    if value != expected:
        raise RawAuthorityRecoveryError(f"recovery plan is stale: {field} changed")


def _revalidate_common(plan: RawAuthorityRecoveryPlan, root: Path, location: ArchiveLocation) -> None:
    if str(root.resolve(strict=False)) != plan.archive_root:
        raise RawAuthorityRecoveryError("recovery plan names a different archive root")
    _same(VERSION_INFO.commit, plan.code_sha, "code SHA")
    _same(_schema_versions(root, location), plan.schema_versions, "schema versions")
    _same(_archive_identity(root, location), plan.archive_identity, "archive identity")
    _same(_digest(plan.archive_identity), plan.archive_identity_digest, "archive identity digest")
    _same(_generation_identity(root, location), plan.active_generation, "active index generation/pointer")
    _same(_file_fingerprint(root / "source.db"), plan.source_fingerprint, "source database")
    _same(_file_fingerprint(location.active_index_path), plan.index_fingerprint, "active index database")
    _same(source_revision_snapshot(root), plan.source_snapshot, "source snapshot")


def _recovery_intent(plan: RawAuthorityRecoveryPlan) -> dict[str, object]:
    return {
        "format": INTENT_FORMAT,
        "operation_id": plan.operation_id,
        "operation": plan.operation,
        "archive_root": plan.archive_root,
        "plan_digest": plan.plan_digest,
        "receipt_path": plan.receipt_path,
        "before_counts": plan.before_counts,
        "candidate_keys": {key: list(value) for key, value in plan.candidate_keys.items()},
        "protected_digest": plan.protected_digest,
        "plan": plan.to_dict(),
    }


def _write_recovery_intent(plan: RawAuthorityRecoveryPlan) -> Path:
    return _write_durable_immutable(
        Path(plan.archive_root),
        _intent_path(Path(plan.receipt_path)),
        _recovery_intent(plan),
        digest_field="intent_sha256",
    )


def _intent_for_plan(plan: RawAuthorityRecoveryPlan) -> dict[str, object] | None:
    path = _intent_path(Path(plan.receipt_path))
    payload = _read_durable_json(Path(plan.archive_root), path)
    if payload is None:
        return None
    expected = payload.get("intent_sha256")
    if not isinstance(expected, str) or expected != _digest(
        {key: value for key, value in payload.items() if key != "intent_sha256"}
    ):
        raise RawAuthorityRecoveryError("existing recovery intent has an invalid self-hash")
    if payload != {**_recovery_intent(plan), "intent_sha256": expected}:
        raise RawAuthorityRecoveryError("existing recovery intent belongs to another operation or plan")
    return payload


def _receipt_payload(
    plan: RawAuthorityRecoveryPlan,
    *,
    before_counts: dict[str, int],
    after_counts: dict[str, int],
    postflight: dict[str, object],
) -> dict[str, object]:
    root = Path(plan.archive_root)
    return {
        "format": RECEIPT_FORMAT,
        "operation_id": plan.operation_id,
        "operation": plan.operation,
        "archive_root": plan.archive_root,
        "plan_digest": plan.plan_digest,
        "code_sha": plan.code_sha,
        "archive_identity": plan.archive_identity,
        "schema_versions": plan.schema_versions,
        "active_generation": plan.active_generation,
        "source_snapshot_before": plan.source_snapshot,
        "source_snapshot_after": source_revision_snapshot(root),
        "source_fingerprint_before": plan.source_fingerprint,
        "index_fingerprint_before": plan.index_fingerprint,
        "source_fingerprint_after": _file_fingerprint(root / "source.db"),
        "index_fingerprint_after": _file_fingerprint(ArchiveLocation.resolve(root).active_index_path),
        "before_counts": before_counts,
        "after_counts": after_counts,
        "candidate_keys": {key: list(value) for key, value in plan.candidate_keys.items()},
        "backup_authority": plan.backup_authority,
        "protected_digest_before": plan.protected_digest,
        "protected_digest_after": postflight["protected_digest"],
        "postflight": postflight,
    }


def _write_recovery_receipt(
    plan: RawAuthorityRecoveryPlan,
    *,
    before_counts: dict[str, int],
    after_counts: dict[str, int],
    postflight: dict[str, object],
) -> Path:
    return _write_durable_immutable(
        Path(plan.archive_root),
        Path(plan.receipt_path),
        _receipt_payload(plan, before_counts=before_counts, after_counts=after_counts, postflight=postflight),
        digest_field="receipt_sha256",
    )


def _committed_postflight(plan: RawAuthorityRecoveryPlan) -> tuple[dict[str, int], dict[str, object]] | None:
    """Return postflight evidence only when an intent's exact mutation committed."""

    root = Path(plan.archive_root)
    location = ArchiveLocation.resolve(root)
    operation = RecoveryOperation(plan.operation)
    excluded = _RESET_TABLES if operation is RecoveryOperation.RESET_CENSUS else _INDEX_TARGETS
    database = root / "source.db" if operation is RecoveryOperation.RESET_CENSUS else location.active_index_path
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        if operation is RecoveryOperation.RESET_CENSUS:
            after_counts = _count_tables(conn, _RESET_TABLES)
            expected_after = dict.fromkeys(_RESET_TABLES, 0)
        else:
            conn.execute("ATTACH DATABASE ? AS src", (str(root / "source.db"),))
            after_counts = _count_tables(conn, _INDEX_TARGETS)
            expected_after = {key: plan.before_counts[key] - len(plan.candidate_keys[key]) for key in _INDEX_TARGETS}
        if after_counts != expected_after:
            return None
        if operation is RecoveryOperation.PRUNE_INDEX_SEEDS:
            if plan.post_target_digest is None:
                raise RawAuthorityRecoveryError("recovery plan is missing its exact index seed post-target digest")
            if _index_seed_digest(conn) != plan.post_target_digest:
                raise RawAuthorityRecoveryError("recovery intent does not match the exact committed index seed state")
        postflight = _postflight(conn, protected_digest=plan.protected_digest, excluded=excluded)
    return after_counts, postflight


def _apply_plan(plan: RawAuthorityRecoveryPlan) -> RawAuthorityRecoveryReport:
    root = Path(plan.archive_root)
    _require_apply_preconditions(root)
    _resolve_receipt_path(root, Path(plan.receipt_path))
    location = ArchiveLocation.resolve(root)
    operation = RecoveryOperation(plan.operation)
    source_db = root / "source.db"
    index_db = location.active_index_path
    excluded = _RESET_TABLES if operation is RecoveryOperation.RESET_CENSUS else _INDEX_TARGETS
    before_counts = dict(plan.before_counts)
    if _intent_for_plan(plan) is not None:
        committed = _committed_postflight(plan)
        if committed is not None:
            after_counts, postflight = committed
            receipt_path = _write_recovery_receipt(
                plan,
                before_counts=before_counts,
                after_counts=after_counts,
                postflight=postflight,
            )
            return RawAuthorityRecoveryReport(
                plan=plan,
                applied=False,
                status="already_satisfied",
                receipt_path=receipt_path,
                after_counts=after_counts,
                postflight=postflight,
            )
    _write_recovery_intent(plan)
    if operation is RecoveryOperation.RESET_CENSUS:
        with closing(sqlite3.connect(source_db)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("BEGIN IMMEDIATE")
            try:
                _revalidate_common(plan, root, location)
                _validate_ledger(conn)
                _validate_integrity(conn, tier="source")
                _same(_count_tables(conn, _RESET_TABLES), before_counts, "census ledger counts")
                _same(_protected_digest(conn, excluded=excluded), plan.protected_digest, "protected source rows")
                _backup_from_plan(plan, connection=conn)
                for table in _RESET_TABLES:
                    conn.execute(f"DELETE FROM {_quote_identifier(table)}")
                after_counts = _count_tables(conn, _RESET_TABLES)
                if any(after_counts.values()):
                    raise RawAuthorityRecoveryError("census reset postflight left ledger rows behind")
                postflight = _postflight(conn, protected_digest=plan.protected_digest, excluded=excluded)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    else:
        with closing(sqlite3.connect(index_db)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("ATTACH DATABASE ? AS src", (str(source_db),))
            conn.execute("BEGIN IMMEDIATE")
            try:
                _revalidate_common(plan, root, location)
                _validate_integrity(conn, tier="active index")
                _same(_index_candidates(conn), plan.candidate_keys, "orphaned index candidate rows")
                _same(_protected_digest(conn, excluded=excluded), plan.protected_digest, "protected index rows")
                _backup_from_plan(plan, connection=conn)
                for key in plan.candidate_keys["raw_revision_heads"]:
                    if (
                        conn.execute("DELETE FROM raw_revision_heads WHERE logical_source_key = ?", (key,)).rowcount
                        != 1
                    ):
                        raise RawAuthorityRecoveryError(f"expected exactly one raw_revision_heads row for {key!r}")
                for key in plan.candidate_keys["raw_revision_applications"]:
                    if (
                        conn.execute("DELETE FROM raw_revision_applications WHERE decision_id = ?", (key,)).rowcount
                        != 1
                    ):
                        raise RawAuthorityRecoveryError(
                            f"expected exactly one raw_revision_applications row for {key!r}"
                        )
                after_counts = {
                    key: int(conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(key)}").fetchone()[0])
                    for key in _INDEX_TARGETS
                }
                expected_after = {key: before_counts[key] - len(plan.candidate_keys[key]) for key in _INDEX_TARGETS}
                if after_counts != expected_after:
                    raise RawAuthorityRecoveryError("index-seed prune postflight changed an unexpected row class")
                if plan.post_target_digest is None or _index_seed_digest(conn) != plan.post_target_digest:
                    raise RawAuthorityRecoveryError("index-seed prune postflight does not match the exact target state")
                postflight = _postflight(conn, protected_digest=plan.protected_digest, excluded=excluded)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    receipt_path = _write_recovery_receipt(
        plan,
        before_counts=before_counts,
        after_counts=after_counts,
        postflight=postflight,
    )
    return RawAuthorityRecoveryReport(
        plan=plan,
        applied=True,
        status="applied",
        receipt_path=receipt_path,
        after_counts=after_counts,
        postflight=postflight,
    )


@dataclass(frozen=True, slots=True)
class _RecoveryArgs:
    archive_root: Path
    operation: RecoveryOperation
    operation_id: str
    expected_plan_digest: str
    backup_manifest: Path | None
    receipt_path: Path


@dataclass(frozen=True, slots=True)
class _RecoveryActuator:
    operation: str
    recovery_operation: RecoveryOperation
    destructive_class: DestructiveClass = "reset"
    required_confirmation: ConfirmationStrength = "confirm_flag"

    def prepare(self, args: _RecoveryArgs) -> MutationPlan:
        live = inspect_raw_authority_recovery(
            args.archive_root,
            args.operation,
            operation_id=args.operation_id,
            backup_manifest=args.backup_manifest,
            receipt_path=args.receipt_path,
        )
        return build_plan(
            operation=self.operation,
            destructive_class=self.destructive_class,
            target_refs=(
                make_target_ref(
                    "source",
                    args.operation.value,
                ),
            ),
            affected_tiers=("source",) if args.operation is RecoveryOperation.RESET_CENSUS else ("index",),
            reversible=False,
            context={"recovery_plan_digest": live.plan_digest, "operation_id": args.operation_id},
        )

    def apply(self, plan: MutationPlan, args: _RecoveryArgs) -> MutationReceipt:
        del plan
        live = inspect_raw_authority_recovery(
            args.archive_root,
            args.operation,
            operation_id=args.operation_id,
            backup_manifest=args.backup_manifest,
            receipt_path=args.receipt_path,
        )
        if live.plan_digest != args.expected_plan_digest:
            raise PlanStaleError("raw-authority recovery plan digest changed before apply")
        report = _apply_plan(live)
        return MutationReceipt(
            operation=self.operation,
            plan_hash=args.expected_plan_digest,
            status="applied" if report.applied else "already_satisfied",
            target_refs=(
                make_target_ref(
                    "source",
                    args.operation.value,
                ),
            ),
            affected_count=sum(report.plan.before_counts.values()),
            detail=None,
            receipt_ref=str(report.receipt_path) if report.receipt_path is not None else None,
            applied_at="recovery",
            domain_receipt=report.to_dict(),
            operation_id=args.operation_id,
        )


class ResetRawAuthorityCensusActuator(_RecoveryActuator):
    def __init__(self) -> None:
        super().__init__("mutate-reset-raw-authority-census", RecoveryOperation.RESET_CENSUS)


class PruneOrphanedIndexRevisionSeedsActuator(_RecoveryActuator):
    def __init__(self) -> None:
        super().__init__("mutate-prune-orphaned-index-revision-seeds", RecoveryOperation.PRUNE_INDEX_SEEDS)


def _receipt_for_plan(plan: RawAuthorityRecoveryPlan) -> dict[str, object] | None:
    path = Path(plan.receipt_path)
    payload = _read_durable_json(Path(plan.archive_root), path)
    if payload is None:
        return None
    expected = payload.get("receipt_sha256")
    if not isinstance(expected, str) or expected != _digest(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    ):
        raise RawAuthorityRecoveryError("existing recovery receipt has an invalid self-hash")
    if payload.get("plan_digest") != plan.plan_digest or payload.get("operation_id") != plan.operation_id:
        raise RawAuthorityRecoveryError("existing recovery receipt belongs to another operation or plan")
    return payload


def _validate_existing_receipt(plan: RawAuthorityRecoveryPlan, receipt: dict[str, object]) -> None:
    root = Path(plan.archive_root)
    location = ArchiveLocation.resolve(root)
    _same(_schema_versions(root, location), plan.schema_versions, "schema versions")
    _same(_archive_identity(root, location), plan.archive_identity, "archive identity")
    _same(_generation_identity(root, location), receipt.get("active_generation"), "active index generation/pointer")
    _same(source_revision_snapshot(root), receipt.get("source_snapshot_after"), "source snapshot")
    _same(_file_fingerprint(root / "source.db"), receipt.get("source_fingerprint_after"), "source database")
    _same(
        _file_fingerprint(location.active_index_path), receipt.get("index_fingerprint_after"), "active index database"
    )
    postflight = receipt.get("postflight")
    if not isinstance(postflight, dict) or receipt.get("protected_digest_after") != postflight.get("protected_digest"):
        raise RawAuthorityRecoveryError("existing recovery receipt has inconsistent postflight evidence")


def _plan_from_intent(
    archive_root: Path,
    operation: RecoveryOperation,
    *,
    operation_id: str,
    receipt_path: Path | None,
) -> RawAuthorityRecoveryPlan:
    root = archive_root.expanduser().resolve(strict=False)
    selected_receipt_path = _resolve_receipt_path(root, receipt_path or _default_receipt_path(root, operation_id))
    intent_path = _intent_path(selected_receipt_path)
    payload = _read_durable_json(root, intent_path)
    if payload is None:
        raise RawAuthorityRecoveryError(f"no restartable recovery intent exists for operation {operation_id!r}")
    serialized_plan = payload.get("plan")
    if not isinstance(serialized_plan, dict):
        raise RawAuthorityRecoveryError("existing recovery intent does not contain a complete recovery plan")
    plan = RawAuthorityRecoveryPlan.from_dict(cast(Mapping[str, object], serialized_plan))
    if plan.archive_root != str(root) or plan.operation != operation.value or plan.operation_id != operation_id:
        raise RawAuthorityRecoveryError("existing recovery intent does not match the requested archive operation")
    _resolve_receipt_path(root, Path(plan.receipt_path))
    _intent_for_plan(plan)
    return plan


def resume_raw_authority_recovery(
    archive_root: Path,
    operation: RecoveryOperation | str,
    *,
    operation_id: str,
    receipt_path: Path | None = None,
) -> RawAuthorityRecoveryReport:
    """Resume a durable intent when the external dry-run plan artifact is unavailable."""

    selected = RecoveryOperation(operation)
    plan = _plan_from_intent(archive_root, selected, operation_id=operation_id, receipt_path=receipt_path)
    return apply_raw_authority_recovery(plan)


def apply_raw_authority_recovery(
    plan: RawAuthorityRecoveryPlan | Path,
    *,
    backup_manifest: Path | None = None,
) -> RawAuthorityRecoveryReport:
    """Apply one exact plan through the named actuator lifecycle."""
    selected = _load_plan(plan) if isinstance(plan, Path) else plan
    _resolve_receipt_path(Path(selected.archive_root), Path(selected.receipt_path))
    if backup_manifest is not None and (
        selected.backup_authority is None
        or str(backup_manifest.resolve(strict=False)) != selected.backup_authority.get("manifest_path")
    ):
        raise RawAuthorityRecoveryError("apply backup manifest does not match the plan authority")
    existing = _receipt_for_plan(selected)
    if existing is not None:
        _require_apply_preconditions(Path(selected.archive_root))
        _validate_existing_receipt(selected, existing)
        return RawAuthorityRecoveryReport(
            plan=selected,
            applied=False,
            status="already_satisfied",
            receipt_path=Path(selected.receipt_path),
            after_counts=cast(dict[str, int], existing.get("after_counts")),
            postflight=cast(dict[str, object], existing.get("postflight")),
        )
    if selected.backup_authority is None:
        raise RawAuthorityRecoveryError("apply requires a dry-run plan with verified backup authority")
    operation = RecoveryOperation(selected.operation)
    root = Path(selected.archive_root)
    args = _RecoveryArgs(
        archive_root=root,
        operation=operation,
        operation_id=selected.operation_id,
        expected_plan_digest=selected.plan_digest,
        backup_manifest=Path(str(selected.backup_authority["manifest_path"])),
        receipt_path=Path(selected.receipt_path),
    )
    actuator: _RecoveryActuator = (
        ResetRawAuthorityCensusActuator()
        if operation is RecoveryOperation.RESET_CENSUS
        else PruneOrphanedIndexRevisionSeedsActuator()
    )
    executor = OperationExecutor()
    try:
        location = ArchiveLocation.resolve(root)
        # A final receipt may be missing after a process crash or I/O failure.
        # Only exact committed postflight evidence can skip a fresh executor
        # authorization. An uncommitted intent is evidence of interruption,
        # not authority to perform the destructive mutation.
        if _intent_for_plan(selected) is not None:
            with OwnedArchiveLocation.acquire(
                location, owner_id=f"raw-authority-recovery:{selected.operation_id}"
            ) as owned:
                current_location = ArchiveLocation.resolve(root)
                assert_owns_archive_location(owned, current_location)
                with RebuildLease(root):
                    if _committed_postflight(selected) is not None:
                        return _apply_plan(selected)
        prepared = executor.prepare(actuator, args)
        if prepared.context.get("recovery_plan_digest") != selected.plan_digest:
            raise PlanStaleError("recovery plan is stale before lease acquisition")
        authorization = executor.authorize(
            actuator,
            prepared,
            actor="cli:maintenance",
            role="maintenance",
            capability="archive.raw_authority_recovery",
            confirmation_strength="confirm_flag",
        )
        with OwnedArchiveLocation.acquire(
            location, owner_id=f"raw-authority-recovery:{selected.operation_id}"
        ) as owned:
            current_location = ArchiveLocation.resolve(root)
            assert_owns_archive_location(owned, current_location)
            with RebuildLease(root):
                result = executor.execute(actuator, prepared, authorization, args)
    except (
        ArchiveLocationError,
        ArchiveOwnershipError,
        FileNotFoundError,
        OSError,
        sqlite3.Error,
        ValueError,
        MutationTransactionError,
    ) as exc:
        if isinstance(exc, RawAuthorityRecoveryError):
            raise
        raise RawAuthorityRecoveryError(str(exc)) from exc
    domain = dict(result.domain_receipt)
    return RawAuthorityRecoveryReport(
        plan=selected,
        applied=result.status == "applied",
        status=result.status,
        receipt_path=Path(str(domain["receipt_path"])) if domain.get("receipt_path") else None,
        after_counts=cast(dict[str, int] | None, domain.get("after_counts")),
        postflight=cast(dict[str, object] | None, domain.get("postflight")),
    )


__all__ = [
    "PLAN_FORMAT",
    "RECEIPT_FORMAT",
    "PruneOrphanedIndexRevisionSeedsActuator",
    "RawAuthorityRecoveryError",
    "RawAuthorityRecoveryPlan",
    "RawAuthorityRecoveryReport",
    "RecoveryOperation",
    "ResetRawAuthorityCensusActuator",
    "apply_raw_authority_recovery",
    "inspect_raw_authority_recovery",
    "resume_raw_authority_recovery",
    "write_recovery_plan",
]

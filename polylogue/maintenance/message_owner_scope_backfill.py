"""Offline compatibility backfill for legacy message assertion ownership.

Message ids are opaque archive identities.  This operation therefore never
tries to recover a session by splitting a message id: the only authority is
the exact ``index.messages(message_id, session_id)`` relation observed in the
active index generation immediately before a rebuild.

The census is read-only and produces an immutable, self-digested plan.  Apply
rechecks that plan after taking exclusive archive ownership, updates only the
provable subset, and emits an immutable self-hashed receipt.  Missing index
owners are retained as typed blockers; malformed or conflicting rows refuse
before the first write.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from polylogue.config import Config
from polylogue.core.hashing import hash_payload
from polylogue.maintenance.offline_guard import running_daemon_pid
from polylogue.paths import render_root
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation, OwnedArchiveLocation
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from polylogue.storage.sqlite.migration_runner import validate_migration_backup_manifest
from polylogue.version import VERSION_INFO

TOOL_VERSION = "message-owner-scope-backfill-v1"
PLAN_FORMAT = "polylogue.message-owner-scope-backfill-plan.v1"
RECEIPT_FORMAT = "polylogue.message-owner-scope-backfill-receipt.v1"
MESSAGE_OWNER_SCOPE_BACKFILL_RECEIPT_ENV = "POLYLOGUE_MESSAGE_OWNER_SCOPE_BACKFILL_RECEIPT"

_ACTIVE_MESSAGE_ASSERTION_COLUMNS = (
    "assertion_id",
    "scope_ref",
    "target_ref",
    "key",
    "kind",
    "value_json",
    "body_text",
    "author_ref",
    "author_kind",
    "evidence_refs_json",
    "status",
    "visibility",
    "confidence",
    "staleness_json",
    "context_policy_json",
    "supersedes_json",
    "created_at_ms",
    "updated_at_ms",
)
_ACTIVE_MESSAGE_ASSERTION_SELECT = """
    SELECT assertion_id, scope_ref, target_ref, key, kind, value_json, body_text,
           author_ref, author_kind, evidence_refs_json, status, visibility,
           confidence, staleness_json, context_policy_json, supersedes_json,
           created_at_ms, updated_at_ms
    FROM assertions
    WHERE target_ref LIKE 'message:%'
      AND kind IN ('mark', 'annotation')
      AND COALESCE(status, 'active') != 'deleted'
    ORDER BY assertion_id
"""


class MessageOwnerScopeBackfillError(RuntimeError):
    """Raised when the ownership backfill cannot proceed safely."""


class MessageOwnerScopeDisposition(StrEnum):
    EXACT_RESOLVABLE = "exact-resolvable"
    ALREADY_SCOPED = "already-scoped"
    MISSING_INDEX_OWNER = "missing-index-owner"
    MALFORMED_SCOPE = "malformed-scope"
    CONFLICTING_SCOPE = "conflicting-scope"


@dataclass(frozen=True, slots=True)
class MessageOwnerScopeBackfillRow:
    assertion_id: str
    target_ref: str
    target_id: str
    kind: str
    scope_ref: str | None
    assertion_snapshot: dict[str, object]
    indexed_owner_ids: tuple[str, ...]
    disposition: MessageOwnerScopeDisposition

    @property
    def exact_owner(self) -> str | None:
        if len(self.indexed_owner_ids) == 1:
            return self.indexed_owner_ids[0]
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "assertion_id": self.assertion_id,
            "target_ref": self.target_ref,
            "target_id": self.target_id,
            "kind": self.kind,
            "scope_ref": self.scope_ref,
            "assertion_snapshot": self.assertion_snapshot,
            "indexed_owner_ids": list(self.indexed_owner_ids),
            "disposition": self.disposition.value,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> MessageOwnerScopeBackfillRow:
        try:
            owners = raw["indexed_owner_ids"]
            if not isinstance(owners, list) or not all(isinstance(item, str) for item in owners):
                raise TypeError("indexed_owner_ids")
            snapshot_raw = raw["assertion_snapshot"]
            if not isinstance(snapshot_raw, dict) or set(snapshot_raw) != set(_ACTIVE_MESSAGE_ASSERTION_COLUMNS):
                raise TypeError("assertion_snapshot")
            snapshot = _assertion_snapshot(cast(Mapping[str, object], snapshot_raw))
            assertion_id = _required_text(raw, "assertion_id")
            target_ref = _required_text(raw, "target_ref")
            kind = _required_text(raw, "kind")
            scope_ref = _optional_text(raw, "scope_ref")
            if (
                snapshot["assertion_id"] != assertion_id
                or snapshot["target_ref"] != target_ref
                or snapshot["kind"] != kind
                or snapshot["scope_ref"] != scope_ref
            ):
                raise TypeError("assertion_snapshot identity")
            return cls(
                assertion_id=assertion_id,
                target_ref=target_ref,
                target_id=_required_text(raw, "target_id"),
                kind=kind,
                scope_ref=scope_ref,
                assertion_snapshot=snapshot,
                indexed_owner_ids=tuple(cast(str, item) for item in owners),
                disposition=MessageOwnerScopeDisposition(_required_text(raw, "disposition")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MessageOwnerScopeBackfillError("message-owner backfill plan has an invalid row") from exc


@dataclass(frozen=True, slots=True)
class MessageOwnerScopeBackfillPlan:
    archive_root: str
    archive_identity: dict[str, object]
    schema_binding: dict[str, int]
    rows: tuple[MessageOwnerScopeBackfillRow, ...]
    plan_digest: str

    @property
    def counts(self) -> dict[str, int]:
        return {
            disposition.value: sum(row.disposition is disposition for row in self.rows)
            for disposition in MessageOwnerScopeDisposition
        }

    @property
    def exact_rows(self) -> tuple[MessageOwnerScopeBackfillRow, ...]:
        return tuple(row for row in self.rows if row.disposition is MessageOwnerScopeDisposition.EXACT_RESOLVABLE)

    @property
    def unresolved_denominator(self) -> int:
        return sum(
            row.disposition
            in {
                MessageOwnerScopeDisposition.MISSING_INDEX_OWNER,
                MessageOwnerScopeDisposition.MALFORMED_SCOPE,
                MessageOwnerScopeDisposition.CONFLICTING_SCOPE,
            }
            for row in self.rows
        )

    def unsigned_payload(self) -> dict[str, object]:
        return {
            "format": PLAN_FORMAT,
            "tool_version": TOOL_VERSION,
            "archive_root": self.archive_root,
            "archive_identity": self.archive_identity,
            "schema_binding": self.schema_binding,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.unsigned_payload(), "plan_digest": self.plan_digest}


@dataclass(frozen=True, slots=True)
class MessageOwnerScopeBackfillReport:
    plan: MessageOwnerScopeBackfillPlan
    after_plan: MessageOwnerScopeBackfillPlan | None
    applied: bool
    terminal_state: str
    updated_count: int
    backup_manifest: Path | None = None
    receipt_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": "apply" if self.applied else "census",
            "applied": self.applied,
            "terminal_state": self.terminal_state,
            "plan_digest": self.plan.plan_digest,
            "counts": self.plan.counts,
            "unresolved_denominator": (
                self.after_plan.unresolved_denominator
                if self.after_plan is not None
                else self.plan.unresolved_denominator
            ),
            "updated_count": self.updated_count,
            "backup_manifest": str(self.backup_manifest) if self.backup_manifest is not None else None,
            "receipt_path": str(self.receipt_path) if self.receipt_path is not None else None,
            "plan": self.plan.to_dict(),
            "after_counts": self.after_plan.counts if self.after_plan is not None else None,
        }


def _required_text(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(key)
    return value


def _optional_text(raw: Mapping[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(key)
    return value


def _assertion_snapshot(raw: Mapping[str, object]) -> dict[str, object]:
    """Return one canonical full durable assertion row, or reject a lossy one."""
    if set(raw) != set(_ACTIVE_MESSAGE_ASSERTION_COLUMNS):
        raise TypeError("assertion snapshot columns")
    snapshot: dict[str, object] = {}
    for column in _ACTIVE_MESSAGE_ASSERTION_COLUMNS:
        value = raw[column]
        if value is not None and (not isinstance(value, (str, int, float)) or isinstance(value, bool)):
            raise TypeError(f"assertion snapshot {column}")
        snapshot[column] = value
    if not isinstance(snapshot["assertion_id"], str) or not snapshot["assertion_id"]:
        raise TypeError("assertion_id")
    if not isinstance(snapshot["target_ref"], str) or not snapshot["target_ref"]:
        raise TypeError("target_ref")
    if not isinstance(snapshot["kind"], str) or not snapshot["kind"]:
        raise TypeError("kind")
    if snapshot["scope_ref"] is not None and not isinstance(snapshot["scope_ref"], str):
        raise TypeError("scope_ref")
    return snapshot


def _active_message_assertion_snapshots(connection: sqlite3.Connection) -> list[dict[str, object]]:
    """Read complete active message assertion rows in their durable order."""
    return [
        _assertion_snapshot(dict(zip(_ACTIVE_MESSAGE_ASSERTION_COLUMNS, row, strict=True)))
        for row in connection.execute(_ACTIVE_MESSAGE_ASSERTION_SELECT).fetchall()
    ]


def _code_sha() -> str:
    return VERSION_INFO.commit or "unknown"


def _schema_version(path: Path) -> int:
    conn: sqlite3.Connection | None = None
    try:
        conn = open_readonly_connection(path)
        row = conn.execute("PRAGMA user_version").fetchone()
    except sqlite3.Error as exc:
        raise MessageOwnerScopeBackfillError(f"could not read schema version from {path}") from exc
    finally:
        if conn is not None:
            conn.close()
    if row is None or not isinstance(row[0], int):
        raise MessageOwnerScopeBackfillError(f"schema version is unavailable for {path}")
    return int(row[0])


def _archive_binding(root: Path) -> tuple[dict[str, object], dict[str, int]]:
    location = ArchiveLocation.resolve(root)
    identity = ArchiveIdentity.resolve_location(location)
    user_path = root / "user.db"
    index_path = location.active_index_path
    schema = {
        "user": _schema_version(user_path),
        "active_index": _schema_version(index_path),
    }
    if schema["user"] != ARCHIVE_VERSION_BY_TIER[ArchiveTier.USER]:
        raise MessageOwnerScopeBackfillError("user.db schema is not current; migrate it before the backfill")
    return (
        {
            "archive_root": str(root.resolve()),
            "authority_identity_digest": identity.authority_identity_digest,
            "durable_id": identity.durable_id,
            "active_index_path": str(index_path.resolve()),
            "active_index_stable_id": identity.tier("index").stable_id,
        },
        schema,
    )


def _classify_row(
    *,
    assertion_id: str,
    target_ref: str,
    target_id: str,
    kind: str,
    scope_ref: str | None,
    assertion_snapshot: dict[str, object],
    indexed_owner_ids: tuple[str, ...],
) -> MessageOwnerScopeBackfillRow:
    scoped_owner: str | None = None
    if scope_ref is not None:
        if scope_ref.startswith("annotation-batch:") and len(scope_ref) > len("annotation-batch:"):
            # Batch scope is immutable annotation provenance, not an owner
            # namespace. The one active-index owner is captured in the final
            # receipt and proved against the replacement candidate.
            if kind != "annotation":
                disposition = MessageOwnerScopeDisposition.MALFORMED_SCOPE
            elif len(indexed_owner_ids) == 0:
                disposition = MessageOwnerScopeDisposition.MISSING_INDEX_OWNER
            elif len(indexed_owner_ids) != 1:
                disposition = MessageOwnerScopeDisposition.CONFLICTING_SCOPE
            else:
                disposition = MessageOwnerScopeDisposition.ALREADY_SCOPED
        elif not scope_ref.startswith("session:") or len(scope_ref) <= len("session:"):
            disposition = MessageOwnerScopeDisposition.MALFORMED_SCOPE
        else:
            # The suffix remains opaque. It is never decoded or split.
            scoped_owner = scope_ref[len("session:") :]
            if len(indexed_owner_ids) == 0:
                disposition = MessageOwnerScopeDisposition.MISSING_INDEX_OWNER
            elif len(indexed_owner_ids) != 1 or indexed_owner_ids[0] != scoped_owner:
                disposition = MessageOwnerScopeDisposition.CONFLICTING_SCOPE
            else:
                disposition = MessageOwnerScopeDisposition.ALREADY_SCOPED
    elif len(indexed_owner_ids) == 1:
        disposition = MessageOwnerScopeDisposition.EXACT_RESOLVABLE
    elif len(indexed_owner_ids) == 0:
        disposition = MessageOwnerScopeDisposition.MISSING_INDEX_OWNER
    else:
        disposition = MessageOwnerScopeDisposition.CONFLICTING_SCOPE
    return MessageOwnerScopeBackfillRow(
        assertion_id=assertion_id,
        target_ref=target_ref,
        target_id=target_id,
        kind=kind,
        scope_ref=scope_ref,
        assertion_snapshot=assertion_snapshot,
        indexed_owner_ids=indexed_owner_ids,
        disposition=disposition,
    )


def _census_connections(root: Path) -> tuple[sqlite3.Connection, sqlite3.Connection]:
    location = ArchiveLocation.resolve(root)
    user: sqlite3.Connection | None = None
    try:
        user = open_readonly_connection(root / "user.db")
        index = open_readonly_connection(location.active_index_path)
    except sqlite3.Error as exc:
        if user is not None:
            user.close()
        raise MessageOwnerScopeBackfillError("could not open user.db and the active index read-only") from exc
    return user, index


def census_message_owner_scope_backfill(archive_root: Path) -> MessageOwnerScopeBackfillPlan:
    """Build a deterministic read-only plan from the active index relation."""
    root = archive_root.resolve()
    if not (root / "user.db").exists():
        raise MessageOwnerScopeBackfillError(f"no user.db at {root / 'user.db'}")
    location = ArchiveLocation.resolve(root)
    if not location.active_index_path.exists():
        raise MessageOwnerScopeBackfillError(f"no active index.db at {location.active_index_path}")
    archive_binding, schema = _archive_binding(root)
    user, index = _census_connections(root)
    try:
        try:
            user_rows = _active_message_assertion_snapshots(user)
            rows: list[MessageOwnerScopeBackfillRow] = []
            for snapshot in user_rows:
                assertion_id = _required_text(snapshot, "assertion_id")
                target_ref = _required_text(snapshot, "target_ref")
                target_id = target_ref[len("message:") :] if target_ref.startswith("message:") else ""
                owners = tuple(
                    sorted(
                        {
                            str(owner)
                            for (owner,) in index.execute(
                                "SELECT DISTINCT session_id FROM messages WHERE message_id = ?",
                                (target_id,),
                            ).fetchall()
                        }
                    )
                )
                rows.append(
                    _classify_row(
                        assertion_id=assertion_id,
                        target_ref=target_ref,
                        target_id=target_id,
                        kind=_required_text(snapshot, "kind"),
                        scope_ref=_optional_text(snapshot, "scope_ref"),
                        assertion_snapshot=snapshot,
                        indexed_owner_ids=owners,
                    )
                )
        except sqlite3.Error as exc:
            raise MessageOwnerScopeBackfillError(
                "could not census message assertions against the active index"
            ) from exc
    finally:
        user.close()
        index.close()
    unsigned = {
        "format": PLAN_FORMAT,
        "tool_version": TOOL_VERSION,
        "archive_root": str(root),
        "archive_identity": archive_binding,
        "schema_binding": schema,
        "rows": [row.to_dict() for row in rows],
    }
    return MessageOwnerScopeBackfillPlan(
        archive_root=str(root),
        archive_identity=archive_binding,
        schema_binding=schema,
        rows=tuple(rows),
        plan_digest=hash_payload(unsigned),
    )


def _write_immutable_json(path: Path, payload: Mapping[str, object], *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise MessageOwnerScopeBackfillError(f"immutable {label} already exists: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor != -1:
            os.close(descriptor)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_message_owner_scope_backfill_plan(plan: MessageOwnerScopeBackfillPlan, path: Path) -> None:
    """Write a plan exactly once, outside the archive, with its digest bound."""
    _write_immutable_json(path, plan.to_dict(), label="message-owner backfill plan")


def _load_plan(path: Path) -> MessageOwnerScopeBackfillPlan:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MessageOwnerScopeBackfillError(f"could not read message-owner backfill plan: {path}") from exc
    if not isinstance(raw, dict) or raw.get("format") != PLAN_FORMAT:
        raise MessageOwnerScopeBackfillError("unsupported message-owner backfill plan format")
    unsigned = dict(raw)
    digest = unsigned.pop("plan_digest", None)
    if not isinstance(digest, str) or hash_payload(unsigned) != digest:
        raise MessageOwnerScopeBackfillError("message-owner backfill plan digest mismatch")
    rows_raw = raw.get("rows")
    archive_identity = raw.get("archive_identity")
    schema_binding = raw.get("schema_binding")
    if not isinstance(rows_raw, list) or not isinstance(archive_identity, dict) or not isinstance(schema_binding, dict):
        raise MessageOwnerScopeBackfillError("message-owner backfill plan has invalid bindings")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in schema_binding.values()):
        raise MessageOwnerScopeBackfillError("message-owner backfill plan has invalid schema binding")
    rows = tuple(
        MessageOwnerScopeBackfillRow.from_dict(cast(Mapping[str, object], row))
        for row in rows_raw
        if isinstance(row, dict)
    )
    if len(rows) != len(rows_raw):
        raise MessageOwnerScopeBackfillError("message-owner backfill plan has an invalid row list")
    return MessageOwnerScopeBackfillPlan(
        archive_root=_required_text(raw, "archive_root"),
        archive_identity=dict(archive_identity),
        schema_binding={str(key): int(value) for key, value in schema_binding.items()},
        rows=rows,
        plan_digest=digest,
    )


def _manifest_identity(path: Path) -> dict[str, object]:
    manifest = path / "manifest.json" if path.is_dir() else path
    data = manifest.read_bytes()
    return {"path": str(manifest.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _durable_message_assertion_state(root: Path) -> dict[str, object]:
    """Fingerprint the durable assertion rows a backfill receipt authorizes."""
    conn: sqlite3.Connection | None = None
    try:
        conn = open_readonly_connection(root / "user.db")
        rows = _active_message_assertion_snapshots(conn)
    except sqlite3.Error as exc:
        raise MessageOwnerScopeBackfillError("could not read durable message assertion state") from exc
    finally:
        if conn is not None:
            conn.close()
    return {"row_count": len(rows), "rows_digest": hash_payload({"rows": rows})}


def _has_active_message_assertions(root: Path) -> bool:
    """Return whether the durable tier needs an index owner relation at all."""
    conn: sqlite3.Connection | None = None
    try:
        conn = open_readonly_connection(root / "user.db")
        row = conn.execute(
            """
            SELECT 1
            FROM assertions
            WHERE target_ref LIKE 'message:%' AND kind IN ('mark', 'annotation')
              AND COALESCE(status, 'active') != 'deleted'
            LIMIT 1
            """
        ).fetchone()
    except sqlite3.Error as exc:
        raise MessageOwnerScopeBackfillError("could not inspect durable message assertions") from exc
    finally:
        if conn is not None:
            conn.close()
    return row is not None


def _expected_after_plan(plan: MessageOwnerScopeBackfillPlan) -> MessageOwnerScopeBackfillPlan:
    """Derive the only durable state a prepared marker can safely recover."""
    rows = tuple(
        MessageOwnerScopeBackfillRow(
            assertion_id=row.assertion_id,
            target_ref=row.target_ref,
            target_id=row.target_id,
            kind=row.kind,
            scope_ref=f"session:{row.exact_owner}",
            assertion_snapshot={**row.assertion_snapshot, "scope_ref": f"session:{row.exact_owner}"},
            indexed_owner_ids=row.indexed_owner_ids,
            disposition=MessageOwnerScopeDisposition.ALREADY_SCOPED,
        )
        if row.disposition is MessageOwnerScopeDisposition.EXACT_RESOLVABLE and row.exact_owner is not None
        else row
        for row in plan.rows
    )
    unsigned = {
        "format": PLAN_FORMAT,
        "tool_version": TOOL_VERSION,
        "archive_root": plan.archive_root,
        "archive_identity": plan.archive_identity,
        "schema_binding": plan.schema_binding,
        "rows": [row.to_dict() for row in rows],
    }
    return MessageOwnerScopeBackfillPlan(
        archive_root=plan.archive_root,
        archive_identity=plan.archive_identity,
        schema_binding=plan.schema_binding,
        rows=rows,
        plan_digest=hash_payload(unsigned),
    )


def _receipt_digest(payload: Mapping[str, object]) -> str:
    return hash_payload({key: value for key, value in payload.items() if key != "receipt_sha256"})


def resolve_message_owner_scope_backfill_receipt_reference(
    archive_root: Path, receipt_path: Path | None = None
) -> Path:
    """Resolve the completed owner-backfill receipt used by rebuild callers."""
    root = Path(archive_root).absolute()
    candidate = receipt_path
    if candidate is None:
        configured = os.environ.get(MESSAGE_OWNER_SCOPE_BACKFILL_RECEIPT_ENV, "").strip()
        if not configured:
            raise MessageOwnerScopeBackfillError(
                "a completed message-owner backfill receipt is required; pass a receipt path or set "
                f"{MESSAGE_OWNER_SCOPE_BACKFILL_RECEIPT_ENV}"
            )
        candidate = Path(configured)
    target = candidate.expanduser().resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return target
    raise MessageOwnerScopeBackfillError(
        "message-owner backfill receipt filename/path must be outside the archive root"
    )


def _owner_bindings(plan: MessageOwnerScopeBackfillPlan) -> list[dict[str, str]]:
    """Bind every complete-plan assertion to the one observed session owner."""
    bindings: list[dict[str, str]] = []
    for row in plan.rows:
        owner = row.exact_owner
        if owner is None:
            raise MessageOwnerScopeBackfillError("complete owner-backfill plan lacks an exact message owner")
        bindings.append(
            {
                "assertion_id": row.assertion_id,
                "target_ref": row.target_ref,
                "owner_session_id": owner,
            }
        )
    return bindings


def _write_prepared_marker(path: Path, *, plan: MessageOwnerScopeBackfillPlan, backup: Mapping[str, object]) -> Path:
    marker = path.with_name(path.name + ".prepared")
    payload: dict[str, object] = {
        "format": RECEIPT_FORMAT,
        "phase": "prepared",
        "tool_version": TOOL_VERSION,
        "code_sha": _code_sha(),
        "plan_digest": plan.plan_digest,
        "archive_identity": plan.archive_identity,
        "schema_binding": plan.schema_binding,
        "before_counts": plan.counts,
        "expected_after_plan_digest": _expected_after_plan(plan).plan_digest,
        "backup_manifest": dict(backup),
        "receipt_path": str(path.resolve()),
        "prepared_at_ms": int(time.time() * 1000),
    }
    payload["receipt_sha256"] = _receipt_digest(payload)
    _write_immutable_json(marker, payload, label="prepared message-owner backfill marker")
    return marker


def _final_receipt(
    *,
    plan: MessageOwnerScopeBackfillPlan,
    after_plan: MessageOwnerScopeBackfillPlan,
    backup: Mapping[str, object],
    receipt_path: Path,
    updated_count: int,
    durable_message_assertion_state: Mapping[str, object],
    recovered_from_prepared: bool = False,
    recovered_receipt_fragment: Mapping[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "format": RECEIPT_FORMAT,
        "phase": "terminal",
        "terminal_state": "committed" if after_plan.unresolved_denominator == 0 else "blocked",
        "complete": after_plan.unresolved_denominator == 0,
        "tool_version": TOOL_VERSION,
        "code_sha": _code_sha(),
        "archive_identity": plan.archive_identity,
        "schema_binding": plan.schema_binding,
        "plan_digest": plan.plan_digest,
        "backup_manifest": dict(backup),
        "before_counts": plan.counts,
        "after_counts": after_plan.counts,
        "after_plan_digest": after_plan.plan_digest,
        "after_owner_bindings": _owner_bindings(after_plan) if after_plan.unresolved_denominator == 0 else [],
        "durable_message_assertion_state": dict(durable_message_assertion_state),
        "updated_count": updated_count,
        "unresolved_denominator": after_plan.unresolved_denominator,
        "recovered_from_prepared": recovered_from_prepared,
        "recovered_receipt_fragment": (
            dict(recovered_receipt_fragment) if recovered_receipt_fragment is not None else None
        ),
        "completed_at_ms": int(time.time() * 1000),
    }
    payload["receipt_sha256"] = _receipt_digest(payload)
    _write_immutable_json(receipt_path, payload, label="message-owner backfill receipt")


def _validate_plan_binding(root: Path, plan: MessageOwnerScopeBackfillPlan) -> MessageOwnerScopeBackfillPlan:
    current = census_message_owner_scope_backfill(root)
    if current.plan_digest != plan.plan_digest:
        raise MessageOwnerScopeBackfillError("message-owner backfill plan changed before apply")
    if current.archive_root != plan.archive_root or current.archive_identity != plan.archive_identity:
        raise MessageOwnerScopeBackfillError("message-owner backfill archive binding changed before apply")
    if current.schema_binding != plan.schema_binding:
        raise MessageOwnerScopeBackfillError("message-owner backfill schema binding changed before apply")
    return current


def _offline_config(root: Path) -> Config:
    return Config(archive_root=root, render_root=render_root(), sources=[])


def _recover_prepared_receipt(
    root: Path,
    *,
    plan: MessageOwnerScopeBackfillPlan,
    backup: Mapping[str, object],
    receipt_path: Path,
) -> MessageOwnerScopeBackfillReport | None:
    """Finalize only a marker whose current durable state is exactly committed."""
    marker = receipt_path.with_name(receipt_path.name + ".prepared")
    if not marker.exists():
        return None
    try:
        raw = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MessageOwnerScopeBackfillError("could not read prepared message-owner backfill marker") from exc
    if not isinstance(raw, dict) or raw.get("format") != RECEIPT_FORMAT or raw.get("phase") != "prepared":
        raise MessageOwnerScopeBackfillError("prepared message-owner backfill marker has unsupported format")
    if raw.get("receipt_sha256") != _receipt_digest(raw):
        raise MessageOwnerScopeBackfillError("prepared message-owner backfill marker content digest mismatch")
    expected_after = _expected_after_plan(plan)
    if (
        raw.get("plan_digest") != plan.plan_digest
        or raw.get("archive_identity") != plan.archive_identity
        or raw.get("schema_binding") != plan.schema_binding
        or raw.get("backup_manifest") != dict(backup)
        or raw.get("receipt_path") != str(receipt_path.resolve())
        or raw.get("expected_after_plan_digest") != expected_after.plan_digest
    ):
        raise MessageOwnerScopeBackfillError("prepared message-owner backfill marker does not match this apply")
    current = census_message_owner_scope_backfill(root)
    if current.plan_digest != expected_after.plan_digest:
        raise MessageOwnerScopeBackfillError(
            "prepared message-owner backfill marker does not prove the committed durable state"
        )
    if receipt_path.exists():
        try:
            terminal = validate_message_owner_scope_backfill_receipt(root, receipt_path)
        except MessageOwnerScopeBackfillError:
            terminal = None
        if terminal is not None:
            if (
                terminal.get("plan_digest") != plan.plan_digest
                or terminal.get("archive_identity") != plan.archive_identity
                or terminal.get("schema_binding") != plan.schema_binding
                or terminal.get("backup_manifest") != dict(backup)
                or terminal.get("after_plan_digest") != expected_after.plan_digest
            ):
                raise MessageOwnerScopeBackfillError(
                    "prepared message-owner backfill marker does not match its terminal receipt"
                )
            updated_count = terminal.get("updated_count")
            if not isinstance(updated_count, int) or isinstance(updated_count, bool):
                raise MessageOwnerScopeBackfillError("prepared terminal receipt has an invalid updated count")
            marker.unlink(missing_ok=True)
            _fsync_directory(marker.parent)
            return MessageOwnerScopeBackfillReport(
                plan=plan,
                after_plan=current,
                applied=True,
                terminal_state="committed",
                updated_count=updated_count,
                backup_manifest=Path(str(backup["path"])),
                receipt_path=receipt_path,
            )
    recovered_receipt_fragment = _preserve_partial_receipt(receipt_path)
    _final_receipt(
        plan=plan,
        after_plan=current,
        backup=backup,
        receipt_path=receipt_path,
        updated_count=len(plan.exact_rows),
        durable_message_assertion_state=_durable_message_assertion_state(root),
        recovered_from_prepared=True,
        recovered_receipt_fragment=recovered_receipt_fragment,
    )
    marker.unlink(missing_ok=True)
    return MessageOwnerScopeBackfillReport(
        plan=plan,
        after_plan=current,
        applied=True,
        terminal_state="committed" if current.unresolved_denominator == 0 else "blocked",
        updated_count=len(plan.exact_rows),
        backup_manifest=Path(str(backup["path"])),
        receipt_path=receipt_path,
    )


def _preserve_partial_receipt(receipt_path: Path) -> dict[str, object] | None:
    """Quarantine a failed terminal publication without overwriting its evidence."""
    if not receipt_path.exists():
        return None
    try:
        raw = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raw = None
    if (
        isinstance(raw, dict)
        and raw.get("format") == RECEIPT_FORMAT
        and raw.get("phase") == "terminal"
        and raw.get("receipt_sha256") == _receipt_digest(raw)
    ):
        raise MessageOwnerScopeBackfillError(
            "prepared message-owner backfill marker coexists with an immutable terminal receipt"
        )
    fragment = receipt_path.with_name(receipt_path.name + ".partial")
    if fragment.exists():
        raise MessageOwnerScopeBackfillError(
            f"prepared message-owner backfill receipt fragment already exists: {fragment}"
        )
    try:
        os.replace(receipt_path, fragment)
        _fsync_directory(receipt_path.parent)
    except OSError as exc:
        raise MessageOwnerScopeBackfillError("could not preserve partial message-owner backfill receipt") from exc
    return _manifest_identity(fragment)


def apply_message_owner_scope_backfill(
    archive_root: Path,
    *,
    plan_path: Path,
    backup_manifest: Path,
    receipt_path: Path,
    dry_run: bool = True,
) -> MessageOwnerScopeBackfillReport:
    """Apply exact legacy owner rows, or return the read-only census by default."""
    root = archive_root.resolve()
    plan = _load_plan(plan_path)
    if plan.archive_root != str(root):
        raise MessageOwnerScopeBackfillError("message-owner backfill plan targets a different archive")
    if dry_run:
        return MessageOwnerScopeBackfillReport(
            plan=plan, after_plan=None, applied=False, terminal_state="census", updated_count=0
        )
    if backup_manifest is None:
        raise MessageOwnerScopeBackfillError(
            "message-owner backfill apply requires a verified user-tier backup manifest"
        )
    receipt_path = resolve_message_owner_scope_backfill_receipt_reference(root, receipt_path)
    marker_path = receipt_path.with_name(receipt_path.name + ".prepared")
    if receipt_path.exists() and not marker_path.exists():
        raise MessageOwnerScopeBackfillError(f"immutable message-owner backfill receipt already exists: {receipt_path}")
    if running_daemon_pid(_offline_config(root)) is not None:
        raise MessageOwnerScopeBackfillError("message-owner backfill requires the daemon to be stopped")

    location = ArchiveLocation.resolve(root)
    try:
        owner = OwnedArchiveLocation.acquire(location, owner_id=f"message-owner-scope-backfill:{os.getpid()}")
    except Exception as exc:
        raise MessageOwnerScopeBackfillError(f"could not acquire exclusive offline archive ownership: {exc}") from exc
    with owner:
        user_path = root / "user.db"
        conn = sqlite3.connect(user_path, timeout=60)
        marker: Path | None = None
        updated_count = 0
        try:
            validate_migration_backup_manifest(backup_manifest, ArchiveTier.USER, connection=conn)
            backup = _manifest_identity(backup_manifest)
            recovered = _recover_prepared_receipt(
                root,
                plan=plan,
                backup=backup,
                receipt_path=receipt_path,
            )
            if recovered is not None:
                return recovered
            current = _validate_plan_binding(root, plan)
            if (
                current.counts[MessageOwnerScopeDisposition.MALFORMED_SCOPE.value]
                or current.counts[MessageOwnerScopeDisposition.CONFLICTING_SCOPE.value]
            ):
                raise MessageOwnerScopeBackfillError(
                    "message-owner backfill refuses malformed or conflicting scope rows"
                )
            marker = _write_prepared_marker(receipt_path, plan=plan, backup=backup)
            conn.execute("BEGIN IMMEDIATE")
            try:
                validate_migration_backup_manifest(backup_manifest, ArchiveTier.USER, connection=conn)
                locked = _validate_plan_binding(root, plan)
                if locked.counts != current.counts or locked.plan_digest != plan.plan_digest:
                    raise MessageOwnerScopeBackfillError("message-owner backfill plan changed under the write lock")
                for row in plan.exact_rows:
                    owner_id = row.exact_owner
                    if owner_id is None:
                        raise MessageOwnerScopeBackfillError("exact-resolvable row has no unique indexed owner")
                    result = conn.execute(
                        """
                        UPDATE assertions
                        SET scope_ref = ?
                        WHERE assertion_id = ? AND scope_ref IS NULL
                          AND target_ref = ? AND kind IN ('mark', 'annotation')
                          AND COALESCE(status, 'active') != 'deleted'
                        """,
                        (f"session:{owner_id}", row.assertion_id, row.target_ref),
                    )
                    if result.rowcount != 1:
                        raise MessageOwnerScopeBackfillError(
                            f"message-owner backfill candidate changed: {row.assertion_id}"
                        )
                    updated_count += 1
                conn.commit()
            except Exception:
                if conn.in_transaction:
                    conn.rollback()
                raise
        except Exception:
            raise
        finally:
            conn.close()
        after = census_message_owner_scope_backfill(root)
        _final_receipt(
            plan=plan,
            after_plan=after,
            backup=backup,
            receipt_path=receipt_path,
            updated_count=updated_count,
            durable_message_assertion_state=_durable_message_assertion_state(root),
        )
        if marker is not None:
            marker.unlink(missing_ok=True)
        return MessageOwnerScopeBackfillReport(
            plan=plan,
            after_plan=after,
            applied=True,
            terminal_state="committed" if after.unresolved_denominator == 0 else "blocked",
            updated_count=updated_count,
            backup_manifest=backup_manifest,
            receipt_path=receipt_path,
        )


def validate_message_owner_scope_backfill_receipt(
    archive_root: Path,
    receipt_path: Path,
    *,
    candidate_index_path: Path | None = None,
) -> dict[str, object]:
    """Validate a complete zero-unresolved receipt for reindex acceptance."""
    try:
        raw = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MessageOwnerScopeBackfillError("could not read message-owner backfill receipt") from exc
    if not isinstance(raw, dict) or raw.get("format") != RECEIPT_FORMAT:
        raise MessageOwnerScopeBackfillError("unsupported message-owner backfill receipt format")
    if raw.get("receipt_sha256") != _receipt_digest(raw):
        raise MessageOwnerScopeBackfillError("message-owner backfill receipt content digest mismatch")
    if raw.get("phase") != "terminal":
        raise MessageOwnerScopeBackfillError("message-owner backfill receipt is not terminal")
    if raw.get("terminal_state") != "committed" or raw.get("complete") is not True:
        raise MessageOwnerScopeBackfillError("message-owner backfill receipt is not complete")
    if raw.get("unresolved_denominator") != 0:
        raise MessageOwnerScopeBackfillError("message-owner backfill receipt has unresolved owners")
    if not isinstance(raw.get("updated_count"), int) or isinstance(raw.get("updated_count"), bool):
        raise MessageOwnerScopeBackfillError("message-owner backfill receipt has an invalid updated count")
    root = archive_root.resolve()
    binding = raw.get("archive_identity")
    if not isinstance(binding, dict) or binding.get("archive_root") != str(root):
        raise MessageOwnerScopeBackfillError("message-owner backfill receipt targets a different archive")
    current_identity = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(root))
    if binding.get("durable_id") != current_identity.durable_id:
        raise MessageOwnerScopeBackfillError("message-owner backfill receipt durable archive binding is stale")
    schema = raw.get("schema_binding")
    if not isinstance(schema, dict) or schema.get("user") != _schema_version(root / "user.db"):
        raise MessageOwnerScopeBackfillError("message-owner backfill receipt user schema binding is stale")
    active_index_path = ArchiveLocation.resolve(root).active_index_path
    if active_index_path.exists() and schema.get("active_index") != _schema_version(active_index_path):
        raise MessageOwnerScopeBackfillError("message-owner backfill receipt active index schema binding is stale")
    if (
        candidate_index_path is not None
        and _schema_version(candidate_index_path) != ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]
    ):
        raise MessageOwnerScopeBackfillError("message-owner backfill receipt candidate index schema is stale")
    durable_state = raw.get("durable_message_assertion_state")
    if not isinstance(durable_state, dict) or durable_state != _durable_message_assertion_state(root):
        raise MessageOwnerScopeBackfillError("message-owner backfill receipt durable message assertion state is stale")
    owner_bindings = _receipt_owner_bindings(root, raw)
    _validate_durable_owner_bindings(root, owner_bindings)
    if candidate_index_path is not None:
        _validate_candidate_message_owners(root, candidate_index_path, owner_bindings=owner_bindings)
    return cast(dict[str, object], raw)


def _receipt_owner_bindings(root: Path, receipt: Mapping[str, object]) -> dict[str, tuple[str, str]]:
    """Load durable assertion-to-owner bindings committed by a complete receipt."""
    raw_bindings = receipt.get("after_owner_bindings")
    if raw_bindings is None:
        # Receipts created before batch provenance was admitted can still prove
        # legacy session-scoped rows from their durable state. Batch-scoped
        # assertions require the explicit binding added with this format.
        conn: sqlite3.Connection | None = None
        try:
            conn = open_readonly_connection(root / "user.db")
            rows = _active_message_assertion_snapshots(conn)
        except sqlite3.Error as exc:
            raise MessageOwnerScopeBackfillError("could not read legacy message-owner receipt bindings") from exc
        finally:
            if conn is not None:
                conn.close()
        bindings: dict[str, tuple[str, str]] = {}
        for row in rows:
            assertion_id = _required_text(row, "assertion_id")
            target_ref = _required_text(row, "target_ref")
            scope_ref = _optional_text(row, "scope_ref")
            if scope_ref is None or not scope_ref.startswith("session:") or len(scope_ref) == len("session:"):
                raise MessageOwnerScopeBackfillError(
                    "legacy message-owner receipt cannot prove non-session-scoped assertions"
                )
            bindings[assertion_id] = (target_ref, scope_ref[len("session:") :])
        return bindings
    if not isinstance(raw_bindings, list):
        raise MessageOwnerScopeBackfillError("message-owner backfill receipt owner bindings are invalid")
    bindings = {}
    for raw_binding in raw_bindings:
        if not isinstance(raw_binding, dict):
            raise MessageOwnerScopeBackfillError("message-owner backfill receipt owner binding is invalid")
        try:
            assertion_id = _required_text(raw_binding, "assertion_id")
            target_ref = _required_text(raw_binding, "target_ref")
            owner_session_id = _required_text(raw_binding, "owner_session_id")
        except (KeyError, TypeError) as exc:
            raise MessageOwnerScopeBackfillError("message-owner backfill receipt owner binding is invalid") from exc
        if assertion_id in bindings:
            raise MessageOwnerScopeBackfillError("message-owner backfill receipt owner bindings are duplicated")
        bindings[assertion_id] = (target_ref, owner_session_id)
    return bindings


def _validate_durable_owner_bindings(root: Path, owner_bindings: Mapping[str, tuple[str, str]]) -> None:
    """Require the receipt's binding set to cover the current durable rows exactly."""
    conn: sqlite3.Connection | None = None
    try:
        conn = open_readonly_connection(root / "user.db")
        rows = conn.execute(
            """
            SELECT assertion_id, target_ref
            FROM assertions
            WHERE target_ref LIKE 'message:%' AND kind IN ('mark', 'annotation')
              AND COALESCE(status, 'active') != 'deleted'
            ORDER BY assertion_id
            """
        ).fetchall()
    except sqlite3.Error as exc:
        raise MessageOwnerScopeBackfillError("could not validate durable message-owner receipt bindings") from exc
    finally:
        if conn is not None:
            conn.close()
    current = {str(assertion_id): str(target_ref) for assertion_id, target_ref in rows}
    if set(owner_bindings) != set(current) or any(owner_bindings[key][0] != target for key, target in current.items()):
        raise MessageOwnerScopeBackfillError("message-owner backfill receipt owner bindings are stale")


def _validate_candidate_message_owners(
    root: Path, candidate_index_path: Path, *, owner_bindings: Mapping[str, tuple[str, str]]
) -> None:
    """Prove each receipt-bound durable assertion has one matching candidate owner."""
    user: sqlite3.Connection | None = None
    index: sqlite3.Connection | None = None
    try:
        user = open_readonly_connection(root / "user.db")
        index = open_readonly_connection(candidate_index_path)
        rows = user.execute(
            """
            SELECT assertion_id, target_ref
            FROM assertions
            WHERE target_ref LIKE 'message:%' AND kind IN ('mark', 'annotation')
              AND COALESCE(status, 'active') != 'deleted'
            ORDER BY assertion_id
            """
        ).fetchall()
        observed_assertion_ids: set[str] = set()
        for assertion_id, target_ref in rows:
            assertion = str(assertion_id)
            target = str(target_ref)
            observed_assertion_ids.add(assertion)
            binding = owner_bindings.get(assertion)
            if binding is None or binding[0] != target:
                raise MessageOwnerScopeBackfillError(
                    f"candidate index has no receipt-bound owner for durable message assertion {assertion}"
                )
            owners = tuple(
                str(owner)
                for (owner,) in index.execute(
                    "SELECT DISTINCT session_id FROM messages WHERE message_id = ? ORDER BY session_id",
                    (target[len("message:") :],),
                )
            )
            if owners != (binding[1],):
                raise MessageOwnerScopeBackfillError(
                    f"candidate index does not own durable message assertion {assertion} for {target}"
                )
        if observed_assertion_ids != set(owner_bindings):
            raise MessageOwnerScopeBackfillError("candidate index receipt-bound assertion set is stale")
    except sqlite3.Error as exc:
        raise MessageOwnerScopeBackfillError("could not validate candidate message ownership") from exc
    finally:
        if user is not None:
            user.close()
        if index is not None:
            index.close()


def validate_message_owner_scope_for_index_replacement(
    archive_root: Path,
    *,
    receipt_path: Path | None,
    candidate_index_path: Path | None = None,
) -> None:
    """Gate promotion on backfilled durable owners and the actual candidate."""
    root = archive_root.resolve()
    active_index_path = ArchiveLocation.resolve(root).active_index_path
    if not active_index_path.exists():
        if _has_active_message_assertions(root):
            if receipt_path is None:
                raise MessageOwnerScopeBackfillError(
                    "message-owner scope backfill complete receipt is required after index reset"
                )
            validate_message_owner_scope_backfill_receipt(
                root,
                receipt_path,
                candidate_index_path=candidate_index_path,
            )
        return
    current = census_message_owner_scope_backfill(root)
    if current.unresolved_denominator:
        raise MessageOwnerScopeBackfillError(
            "message-owner scope backfill is incomplete against the current active index"
        )
    if current.exact_rows:
        raise MessageOwnerScopeBackfillError("message-owner scope backfill must complete before index replacement")
    if current.rows and receipt_path is None:
        raise MessageOwnerScopeBackfillError(
            "message-owner scope backfill complete receipt is required after backfill before index replacement"
        )
    if receipt_path is not None:
        validate_message_owner_scope_backfill_receipt(
            root,
            receipt_path,
            candidate_index_path=candidate_index_path,
        )
    elif candidate_index_path is not None:
        _validate_candidate_message_owners(root, candidate_index_path, owner_bindings={})


__all__ = [
    "MessageOwnerScopeBackfillError",
    "MessageOwnerScopeBackfillPlan",
    "MessageOwnerScopeBackfillReport",
    "MessageOwnerScopeBackfillRow",
    "MessageOwnerScopeDisposition",
    "MESSAGE_OWNER_SCOPE_BACKFILL_RECEIPT_ENV",
    "PLAN_FORMAT",
    "RECEIPT_FORMAT",
    "TOOL_VERSION",
    "apply_message_owner_scope_backfill",
    "census_message_owner_scope_backfill",
    "resolve_message_owner_scope_backfill_receipt_reference",
    "validate_message_owner_scope_backfill_receipt",
    "validate_message_owner_scope_for_index_replacement",
    "write_message_owner_scope_backfill_plan",
]

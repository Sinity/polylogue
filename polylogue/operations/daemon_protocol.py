"""Typed archive-scoped daemon operation envelopes.

The envelope deliberately carries readiness and authority with the result so
clients do not need a health/probe request before every operation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from polylogue.core.enums import OperationStatus

DAEMON_OPERATION_PROTOCOL = "polylogue.daemon-operation/v1"
MAX_OPERATION_BODY_BYTES = 64 * 1024
MAX_OPERATION_RESULT_BYTES = 8 * 1024 * 1024


class DaemonAuthority(StrEnum):
    READ = "read"
    WRITE = "write"
    CONTROL = "control"
    LONG_RUNNING = "long-running"


class DaemonFallback(StrEnum):
    DIRECT_READ = "direct-read"
    NEVER = "never"


class DaemonOperationOutcome(StrEnum):
    """Lifecycle outcomes for one machine exchange."""

    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed-out"
    DISCONNECTED_BEFORE_ACCEPTANCE = "disconnected-before-acceptance"
    DISCONNECTED_AFTER_ACCEPTANCE = "disconnected-after-acceptance"
    INDETERMINATE = "indeterminate"
    RESTARTED = "restarted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class AcceptedOperationReference:
    """Immutable reference returned after durable admission of long work."""

    operation_id: str
    accepted_at: str
    status_operation: str = "operation.status"

    def to_dict(self) -> dict[str, str]:
        return {
            "operation_id": self.operation_id,
            "accepted_at": self.accepted_at,
            "status_operation": self.status_operation,
        }


@dataclass(frozen=True, slots=True)
class AuthoritySnapshot:
    """Coherent authority evidence attached to every result."""

    archive_identity: str
    generation: str
    schema_versions: dict[str, int]
    served_by: str
    elapsed_ms: int = 0
    queue_ms: int = 0
    degraded_components: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_identity": self.archive_identity,
            "generation": self.generation,
            "schema_versions": dict(self.schema_versions),
            "served_by": self.served_by,
            "elapsed_ms": self.elapsed_ms,
            "queue_ms": self.queue_ms,
            "degraded_components": list(self.degraded_components),
        }


@dataclass(frozen=True, slots=True)
class DaemonOperationError:
    code: str
    detail: str
    retryable: bool = False
    outcome: DaemonOperationOutcome = DaemonOperationOutcome.FAILED

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "detail": self.detail,
            "retryable": self.retryable,
            "outcome": self.outcome.value,
        }


@dataclass(frozen=True, slots=True)
class DaemonOperationSpec:
    """The machine operation authority shared by every daemon surface.

    This is deliberately metadata only.  Implementations remain in the
    canonical operation/facade layer and adapters only serialize this shape.
    """

    name: str
    authority: DaemonAuthority
    fallback: DaemonFallback
    capability: str = "read"
    deadline_s: float = 2.0
    cancellable: bool = True
    progress: bool = False
    accepted_reference: bool = False
    request_contract: str = "object"
    result_contract: str = "object"
    max_body_bytes: int = MAX_OPERATION_BODY_BYTES
    """Bound on this operation's request body; a selection carries more than a
    parameter map, so the limit belongs to the operation, not the transport."""
    error_contract: str = "polylogue.daemon-error/v1"
    authority_metadata: tuple[str, ...] = (
        "archive",
        "generation",
        "served_by",
        "elapsed_ms",
        "queue_ms",
        "degraded_components",
    )
    request_type: str = ""
    result_type: str = ""
    idempotent: bool = False
    cancellation_outcomes: tuple[str, ...] = (
        "cancelled",
        "timed-out",
        "disconnected-before-acceptance",
        "disconnected-after-acceptance",
        "indeterminate",
    )

    def __post_init__(self) -> None:
        if self.request_type and self.result_type:
            return
        stem = "".join(part.capitalize() for part in self.name.replace(".", "-").split("-"))
        object.__setattr__(self, "request_type", self.request_type or f"{stem}Request")
        object.__setattr__(self, "result_type", self.result_type or f"{stem}Result")

    @property
    def direct_allowed(self) -> bool:
        return self.fallback is DaemonFallback.DIRECT_READ

    def to_dict(self) -> dict[str, object]:
        """Serialize the declaration for discovery and conformance checks."""
        return {
            "name": self.name,
            "authority": self.authority.value,
            "fallback": self.fallback.value,
            "capability": self.capability,
            "deadline_s": self.deadline_s,
            "cancellable": self.cancellable,
            "progress": self.progress,
            "accepted_reference": self.accepted_reference,
            "request_contract": self.request_contract,
            "result_contract": self.result_contract,
            "max_body_bytes": self.max_body_bytes,
            "error_contract": self.error_contract,
            "authority_metadata": list(self.authority_metadata),
            "request_type": self.request_type,
            "result_type": self.result_type,
            "idempotent": self.idempotent,
            "cancellation_outcomes": list(self.cancellation_outcomes),
        }


DAEMON_OPERATION_SPECS: tuple[DaemonOperationSpec, ...] = (
    DaemonOperationSpec(
        "cli.query",
        DaemonAuthority.READ,
        DaemonFallback.DIRECT_READ,
        result_contract="cli.query.result/v1",
        request_type="QueryRequest",
        result_type="QueryResult",
    ),
    DaemonOperationSpec(
        "query.units",
        DaemonAuthority.READ,
        DaemonFallback.DIRECT_READ,
        result_contract="query.units.result/v1",
        request_type="QueryUnitsRequest",
        result_type="QueryUnitsResult",
    ),
    DaemonOperationSpec(
        "status",
        DaemonAuthority.READ,
        DaemonFallback.DIRECT_READ,
        result_contract="status.result/v1",
        request_type="StatusRequest",
        result_type="StatusResult",
    ),
    DaemonOperationSpec(
        "completion",
        DaemonAuthority.READ,
        DaemonFallback.DIRECT_READ,
        result_contract="completion.result/v1",
        request_type="CompletionRequest",
        result_type="CompletionResult",
    ),
    DaemonOperationSpec(
        "facets",
        DaemonAuthority.READ,
        DaemonFallback.DIRECT_READ,
        result_contract="facets.result/v1",
        request_type="FacetsRequest",
        result_type="FacetsResult",
    ),
    DaemonOperationSpec(
        "ingest",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.ingest",
        deadline_s=300.0,
        progress=True,
        accepted_reference=True,
        request_contract="ingest.request/v1",
        result_contract="ingest.result/v1",
        request_type="IngestRequest",
        result_type="IngestResult",
    ),
    DaemonOperationSpec(
        "mutation.session.delete.preview",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.delete_session",
        deadline_s=30.0,
        # A preview carries the exact selection: up to
        # ``DELETE_PREVIEW_MAX_SESSION_IDS`` session ids, not a parameter map.
        max_body_bytes=64 * 1024 * 1024,
        request_contract="mutation.session.delete.preview.request/v1",
        result_contract="mutation.session.delete.preview.result/v1",
    ),
    DaemonOperationSpec(
        "mutation.session.delete.authorize",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.delete_session",
        deadline_s=30.0,
        request_contract="mutation.session.delete.authorize.request/v1",
        result_contract="mutation.session.delete.authorize.result/v1",
    ),
    DaemonOperationSpec(
        "mutation.session.delete.cancel",
        DaemonAuthority.CONTROL,
        DaemonFallback.NEVER,
        capability="archive.delete_session",
        deadline_s=30.0,
        request_contract="mutation.session.delete.cancel.request/v1",
        result_contract="mutation.session.delete.cancel.result/v1",
    ),
    DaemonOperationSpec(
        "mutation.session.delete.execute",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.delete_session",
        deadline_s=300.0,
        progress=True,
        accepted_reference=True,
        request_contract="mutation.session.delete.execute.request/v1",
        result_contract="mutation.result/v1",
    ),
    DaemonOperationSpec(
        "mutation.session.tag",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        max_body_bytes=64 * 1024 * 1024,
        capability="archive.bulk_tag_sessions",
        deadline_s=120.0,
        request_contract="mutation.session.tag.request/v1",
        result_contract="mutation.result/v1",
    ),
    DaemonOperationSpec(
        "mutation.session.metadata",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        max_body_bytes=64 * 1024 * 1024,
        capability="archive.set_metadata",
        deadline_s=120.0,
        request_contract="mutation.session.metadata.request/v1",
        result_contract="mutation.result/v1",
    ),
)

MAX_DECLARED_OPERATION_BODY_BYTES: int = max(spec.max_body_bytes for spec in DAEMON_OPERATION_SPECS)
"""Transport-level read bound: the operation is only known after parsing, so
the socket read is capped by the largest declared body and the operation's own
``max_body_bytes`` is enforced once the request names it."""

MUTATION_OPERATION_NAMES: frozenset[str] = frozenset(
    spec.name for spec in DAEMON_OPERATION_SPECS if spec.authority is not DaemonAuthority.READ
)
"""Operations a CLI adapter may never execute in its own process."""

if len({spec.name for spec in DAEMON_OPERATION_SPECS}) != len(DAEMON_OPERATION_SPECS):
    raise RuntimeError("daemon operation names must be unique")

DAEMON_OPERATION_SCHEMA: dict[str, dict[str, object]] = {spec.name: spec.to_dict() for spec in DAEMON_OPERATION_SPECS}


def daemon_operation_schema() -> dict[str, dict[str, object]]:
    """Return a defensive copy used by clients, server discovery and tests."""
    return {name: dict(metadata) for name, metadata in DAEMON_OPERATION_SCHEMA.items()}


def daemon_operation_spec(name: str) -> DaemonOperationSpec | None:
    return next((spec for spec in DAEMON_OPERATION_SPECS if spec.name == name), None)


@dataclass(frozen=True, slots=True)
class DaemonOperationRequest:
    operation: str
    payload: dict[str, object]
    archive_root: str | None = None
    index_schema_version: int | None = None
    daemon_version: str | None = None
    request_id: str | None = None
    deadline_ms: int | None = None
    idempotency_key: str | None = None
    cancellation_token: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> DaemonOperationRequest:
        protocol = raw.get("protocol")
        operation = raw.get("operation")
        payload = raw.get("payload", {})
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        if protocol != DAEMON_OPERATION_PROTOCOL:
            raise ValueError("unsupported daemon operation protocol")
        if daemon_operation_spec(operation) is None:
            raise ValueError(f"operation is not declared: {operation}")
        archive_root = raw.get("archive_root")
        schema = raw.get("index_schema_version")
        version = raw.get("daemon_version")
        request_id = raw.get("request_id")
        deadline_ms = raw.get("deadline_ms")
        idempotency_key = raw.get("idempotency_key")
        cancellation_token = raw.get("cancellation_token")
        if archive_root is not None and not isinstance(archive_root, str):
            raise ValueError("archive_root must be a string")
        if schema is not None and (not isinstance(schema, int) or isinstance(schema, bool)):
            raise ValueError("index_schema_version must be an integer")
        if version is not None and not isinstance(version, str):
            raise ValueError("daemon_version must be a string")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if deadline_ms is not None and (
            not isinstance(deadline_ms, int) or isinstance(deadline_ms, bool) or deadline_ms <= 0
        ):
            raise ValueError("deadline_ms must be a positive integer")
        if idempotency_key is not None and (not isinstance(idempotency_key, str) or not idempotency_key.strip()):
            raise ValueError("idempotency_key must be a non-empty string")
        if cancellation_token is not None and (
            not isinstance(cancellation_token, str) or not cancellation_token.strip()
        ):
            raise ValueError("cancellation_token must be a non-empty string")
        return cls(
            operation.strip(),
            dict(payload),
            archive_root,
            schema,
            version,
            request_id,
            deadline_ms,
            idempotency_key,
            cancellation_token,
        )

    @property
    def fingerprint(self) -> str:
        """Stable exchange identity used for safe duplicate recovery."""
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "protocol": DAEMON_OPERATION_PROTOCOL,
            "operation": self.operation,
            "payload": self.payload,
            "archive_root": self.archive_root,
            "index_schema_version": self.index_schema_version,
            "daemon_version": self.daemon_version,
            "request_id": self.request_id,
            "deadline_ms": self.deadline_ms,
        }
        if self.idempotency_key is not None:
            result["idempotency_key"] = self.idempotency_key
        if self.cancellation_token is not None:
            result["cancellation_token"] = self.cancellation_token
        return result


@dataclass(frozen=True, slots=True)
class DaemonOperationEnvelope:
    operation: str
    archive: dict[str, object]
    generation: dict[str, object]
    readiness: dict[str, object]
    authority: dict[str, object]
    progress: dict[str, object]
    outcome: OperationStatus | str = OperationStatus.COMPLETED
    served_by: dict[str, object] | None = None
    timing: dict[str, object] | None = None
    degraded_components: tuple[str, ...] = ()
    schema_versions: dict[str, int] | None = None
    result: object = None
    error: dict[str, object] | None = None
    request_id: str | None = None
    accepted_reference: dict[str, object] | None = None
    authority_snapshot: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": DAEMON_OPERATION_PROTOCOL,
            "operation": self.operation,
            "archive": self.archive,
            "generation": self.generation,
            "readiness": self.readiness,
            "authority": self.authority,
            "progress": self.progress,
            "outcome": self.outcome.value if isinstance(self.outcome, OperationStatus) else self.outcome,
            "served_by": self.served_by or {},
            "timing": self.timing or {},
            "degraded_components": list(self.degraded_components),
            "schema_versions": self.schema_versions or {},
            "result": self.result,
            "error": self.error,
            "request_id": self.request_id,
            "accepted_reference": self.accepted_reference,
            "authority_snapshot": self.authority_snapshot,
        }


def archive_identity(
    archive_root: Path, *, schema_version: int, daemon_version: str
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Build an identity/readiness projection without a separate probe."""

    index_path = archive_root / "index.db"
    try:
        stat = index_path.stat()
    except OSError:
        stat = None
    generation: dict[str, object] = {
        "index_schema_version": schema_version,
        "index_size_bytes": stat.st_size if stat is not None else 0,
        "index_mtime_ns": stat.st_mtime_ns if stat is not None else None,
    }
    generation["id"] = hashlib.sha256(
        f"{archive_root.resolve()}:{generation['index_schema_version']}:{generation['index_size_bytes']}:{generation['index_mtime_ns']}".encode()
    ).hexdigest()[:32]
    ready = stat is not None
    archive: dict[str, object] = {
        "root": str(archive_root),
        "daemon_version": daemon_version,
        "index_schema_version": schema_version,
        "archive_identity": str(archive_root.resolve()),
    }
    readiness: dict[str, object] = {
        "state": "ready" if ready else "unavailable",
        "ready": ready,
        "reason": None if ready else "index_missing",
    }
    return archive, generation, readiness


__all__ = [
    "DAEMON_OPERATION_PROTOCOL",
    "DAEMON_OPERATION_SPECS",
    "MUTATION_OPERATION_NAMES",
    "MAX_DECLARED_OPERATION_BODY_BYTES",
    "MAX_OPERATION_BODY_BYTES",
    "MAX_OPERATION_RESULT_BYTES",
    "DaemonAuthority",
    "DaemonOperationOutcome",
    "DaemonFallback",
    "DaemonOperationSpec",
    "DaemonOperationEnvelope",
    "DaemonOperationRequest",
    "AcceptedOperationReference",
    "AuthoritySnapshot",
    "DaemonOperationError",
    "OperationStatus",
    "archive_identity",
    "daemon_operation_spec",
    "daemon_operation_schema",
    "DAEMON_OPERATION_SCHEMA",
]

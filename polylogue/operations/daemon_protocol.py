"""Typed archive-scoped daemon operation envelopes.

The envelope deliberately carries readiness and authority with the result so
clients do not need a health/probe request before every operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

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


class DaemonOutcome(StrEnum):
    COMPLETE = "complete"
    ACCEPTED = "accepted"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INDETERMINATE = "indeterminate"


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
    error_contract: str = "polylogue.daemon-error/v1"
    authority_metadata: tuple[str, ...] = (
        "archive",
        "generation",
        "served_by",
        "elapsed_ms",
        "queue_ms",
        "degraded_components",
    )

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
            "error_contract": self.error_contract,
            "authority_metadata": list(self.authority_metadata),
        }


DAEMON_OPERATION_SPECS: tuple[DaemonOperationSpec, ...] = (
    DaemonOperationSpec(
        "cli.query", DaemonAuthority.READ, DaemonFallback.DIRECT_READ, result_contract="cli.query.result/v1"
    ),
    DaemonOperationSpec(
        "query.units", DaemonAuthority.READ, DaemonFallback.DIRECT_READ, result_contract="query.units.result/v1"
    ),
    DaemonOperationSpec("status", DaemonAuthority.READ, DaemonFallback.DIRECT_READ, result_contract="status.result/v1"),
    DaemonOperationSpec(
        "completion", DaemonAuthority.READ, DaemonFallback.DIRECT_READ, result_contract="completion.result/v1"
    ),
    DaemonOperationSpec("facets", DaemonAuthority.READ, DaemonFallback.DIRECT_READ, result_contract="facets.result/v1"),
)

if len({spec.name for spec in DAEMON_OPERATION_SPECS}) != len(DAEMON_OPERATION_SPECS):
    raise RuntimeError("daemon operation names must be unique")


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
        archive_root = raw.get("archive_root")
        schema = raw.get("index_schema_version")
        version = raw.get("daemon_version")
        request_id = raw.get("request_id")
        deadline_ms = raw.get("deadline_ms")
        if archive_root is not None and not isinstance(archive_root, str):
            raise ValueError("archive_root must be a string")
        if schema is not None and not isinstance(schema, int):
            raise ValueError("index_schema_version must be an integer")
        if version is not None and not isinstance(version, str):
            raise ValueError("daemon_version must be a string")
        if request_id is not None and (not isinstance(request_id, str) or not request_id.strip()):
            raise ValueError("request_id must be a non-empty string")
        if deadline_ms is not None and (not isinstance(deadline_ms, int) or deadline_ms <= 0):
            raise ValueError("deadline_ms must be a positive integer")
        return cls(operation.strip(), dict(payload), archive_root, schema, version, request_id, deadline_ms)

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": DAEMON_OPERATION_PROTOCOL,
            "operation": self.operation,
            "payload": self.payload,
            "archive_root": self.archive_root,
            "index_schema_version": self.index_schema_version,
            "daemon_version": self.daemon_version,
            "request_id": self.request_id,
            "deadline_ms": self.deadline_ms,
        }


@dataclass(frozen=True, slots=True)
class DaemonOperationEnvelope:
    operation: str
    archive: dict[str, object]
    generation: dict[str, object]
    readiness: dict[str, object]
    authority: dict[str, object]
    progress: dict[str, object]
    outcome: DaemonOutcome | str = DaemonOutcome.COMPLETE
    served_by: dict[str, object] | None = None
    timing: dict[str, object] | None = None
    degraded_components: tuple[str, ...] = ()
    schema_versions: dict[str, int] | None = None
    result: object = None
    error: dict[str, object] | None = None
    request_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": DAEMON_OPERATION_PROTOCOL,
            "operation": self.operation,
            "archive": self.archive,
            "generation": self.generation,
            "readiness": self.readiness,
            "authority": self.authority,
            "progress": self.progress,
            "outcome": self.outcome.value if isinstance(self.outcome, DaemonOutcome) else self.outcome,
            "served_by": self.served_by or {},
            "timing": self.timing or {},
            "degraded_components": list(self.degraded_components),
            "schema_versions": self.schema_versions or {},
            "result": self.result,
            "error": self.error,
            "request_id": self.request_id,
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
    "MAX_OPERATION_BODY_BYTES",
    "MAX_OPERATION_RESULT_BYTES",
    "DaemonAuthority",
    "DaemonFallback",
    "DaemonOperationSpec",
    "DaemonOperationEnvelope",
    "DaemonOutcome",
    "DaemonOperationRequest",
    "archive_identity",
    "daemon_operation_spec",
]

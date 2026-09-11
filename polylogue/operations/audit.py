"""Durable audit.db repository for mutation authority and lifecycle evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from polylogue.operations.mutation_transaction import (
    AuthorizationMismatchError,
    MutationAuthorization,
    MutationPlan,
    MutationPreview,
    MutationPrincipal,
    MutationReceipt,
    MutationTarget,
    RecoveryDisposition,
    RecoveryOperation,
    RecoveryTargetDisposition,
    TokenConsumedError,
    TokenExpiredError,
    validate_mutation_plan_integrity,
)
from polylogue.storage.sqlite.archive_tiers.source_items import FrozenSourceManifest
from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator, AuditMutation
from polylogue.storage.sqlite.audit_continuity import AuditContinuityPendingError as AuditContinuityPendingError
from polylogue.storage.sqlite.audit_leaf import (
    AuditLeafError,
    assert_verified_audit_leaf,
    open_verified_audit_connection,
)

AuditTargetState = Literal[
    "pending",
    "running",
    "applied",
    "already_satisfied",
    "rejected",
    "failed",
    "unknown",
    "acknowledged",
    "cancelled",
]
_F = TypeVar("_F", bound=Callable[..., object])
_CONFIRMATION_STRENGTH_ORDER = {"role_only": 0, "confirm_flag": 1, "bound_token": 2}


@dataclass(frozen=True, slots=True)
class MachineRequestBinding:
    """Authenticated exchange identity, without request content or credentials."""

    archive_identity: str
    request_id: str
    principal_ref: str
    fingerprint: str
    operation_name: str

    def __post_init__(self) -> None:
        if not all((self.archive_identity, self.request_id, self.principal_ref, self.operation_name)):
            raise ValueError("machine request binding requires archive, request, principal and operation")
        if len(self.fingerprint) != 64 or any(c not in "0123456789abcdef" for c in self.fingerprint):
            raise ValueError("machine request fingerprint must be a SHA-256 digest")

    def to_dict(self) -> dict[str, str]:
        return {field.name: str(getattr(self, field.name)) for field in fields(self)}


class MachineRequestConflictError(ValueError):
    """An exchange identifier was already bound to different authenticated intent."""


class MachineRequestRecoveredError(RuntimeError):
    """Stop dispatch when a durable exchange already owns the domain effect."""

    def __init__(self, record: dict[str, object]) -> None:
        super().__init__("machine request already has a durable domain reference")
        self.record = record


#: Terminal reasons that deliberately keep a finished run open to bounded
#: operator adjudication.  ``recovery_unknown`` is the wedge a blocked
#: classification installs; ``recovered_applied`` is the duplicate-effect
#: barrier a confirmed-applied classification installs.  A partial
#: classification keeps its declared continuation visible.  Neither may become
#: unrecoverable: without an adjudication route a single misclassification
#: would permanently refuse every later attempt on the same targets.
_ADJUDICABLE_TERMINAL_REASONS = frozenset({"recovery_unknown", "recovered_applied"})


def _is_adjudicable_recovery(status: str, terminal_reason: str | None) -> bool:
    """Return whether a run still carries unresolved recovery authority."""

    reason = terminal_reason or ""
    return status in {"running", "interrupted"} or (
        status in {"failed", "completed"}
        and (reason in _ADJUDICABLE_TERMINAL_REASONS or reason.startswith("recovered_partial:"))
    )


def _run_state_for_targets(states: list[str]) -> tuple[str, str | None]:
    """Derive the parent lifecycle state from the complete target set."""

    if not states:
        return "completed", None
    if "unknown" in states:
        return "interrupted", "unknown_effect"
    if "rejected" in states:
        return "failed", "target_rejected"
    if "failed" in states:
        return "failed", "domain_failure"
    if states and all(state in {"applied", "already_satisfied"} for state in states):
        return "completed", None
    return "running", None


def _receipt_event_detail(receipt: MutationReceipt | None, *, status: str, reason: str | None) -> dict[str, object]:
    """Return bounded audit evidence without copying user-authored domain payloads."""

    return {
        "status": status,
        "reason": (reason or "")[:512],
        "receipt_ref": None if receipt is None else receipt.receipt_ref,
        "target_count": 0 if receipt is None else len(receipt.target_refs),
        "affected_count": 0 if receipt is None else receipt.affected_count,
    }


def token_sha256(token: str) -> str:
    """Return the only representation of a bearer token accepted for storage."""

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _linux_process_start_ticks(pid: int) -> str | None:
    """Return Linux /proc start ticks without misparsing a spaced process name."""

    try:
        _prefix, delimiter, suffix = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rpartition(")")
        if not delimiter:
            return None
        return suffix.split()[19]
    except (IndexError, OSError):
        return None


def _attempt_owner_liveness(owner_id: str | None) -> Literal["live", "dead", "unknown"]:
    """Classify an owner without mistaking unavailable liveness evidence for death."""

    if owner_id is None:
        return "unknown"
    parts = owner_id.split(":")
    if len(parts) not in {2, 3} or parts[0] != "pid":
        return "unknown"
    try:
        pid = int(parts[1])
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    except (OSError, ValueError):
        return "unknown"
    if len(parts) == 2:
        return "live"
    start_ticks = _linux_process_start_ticks(pid)
    if start_ticks is None:
        return "unknown"
    return "live" if start_ticks == parts[2] else "dead"


def _current_process_attempt_owner() -> str:
    """Return a local process identity that rejects PID reuse when available."""

    pid = os.getpid()
    start_ticks = _linux_process_start_ticks(pid)
    if start_ticks is None:
        return f"pid:{pid}"
    return f"pid:{pid}:{start_ticks}"


def _attempt_owner_is_live(owner_id: str | None) -> bool:
    """Return whether an attempt's recorded local process is still its owner."""

    return _attempt_owner_liveness(owner_id) == "live"


@dataclass(frozen=True, slots=True)
class _StoredAuthorizationDigest:
    """A persisted digest available only while replaying a continuity command."""

    value: str


def _continuity_mutation(kind: str) -> Callable[[_F], _F]:
    """Route one audit repository state transition through the source WAL."""

    def decorate(method: _F) -> _F:
        @wraps(method)
        def wrapped(self: AuditRepository, *args: object, **kwargs: object) -> object:
            # The audit tier can be upgraded before source.db installs its
            # matching WAL table. Keep that release window operational; the
            # coordinator becomes mandatory as soon as both schema halves are
            # present.
            if not self._continuity.is_available():
                if self._machine_binding is not None:
                    raise RuntimeError("machine acceptance requires source-WAL audit continuity")
                return method(self, *args, **kwargs)
            payload = self._continuity_payload(kind, args, kwargs)
            if self._machine_binding is not None and self._machine_binding[1] == kind:
                binding = self._machine_binding[0]
                prior = self.machine_request(binding)
                if prior is not None and self._machine_part is None:
                    raise MachineRequestRecoveredError(prior)
                payload["machine_request"] = binding.to_dict()
                if self._machine_part is not None:
                    payload["machine_part"] = self._machine_part
                if self._machine_deadline_unix_ms is not None:
                    payload["accepted_deadline_unix_ms"] = self._machine_deadline_unix_ms
                if prior is None and self._before_machine_prepare is not None:
                    self._before_machine_prepare()
            mutation = AuditMutation(
                kind=kind,
                mutation_id=f"audit-mutation:{secrets.token_urlsafe(18)}",
                created_at_ms=int(time.time() * 1000),
                payload=payload,
            )

            def apply(conn: sqlite3.Connection, _mutation: AuditMutation) -> object:
                self._coordinated_connection = conn
                self._coordinated_mutation = _mutation
                try:
                    result = method(self, *args, **kwargs)
                    self._bind_machine_result(conn, _mutation, result)
                    return result
                finally:
                    self._coordinated_mutation = None
                    self._coordinated_connection = None

            result = self._continuity.execute(mutation, apply)
            if self._on_commit is not None:
                self._on_commit()
            return result

        return cast(_F, wrapped)

    return decorate


def _target_from_payload(raw: object) -> MutationTarget:
    value = cast(dict[str, object], raw)
    return MutationTarget(
        kind=cast(str, value["kind"]),
        ref=cast(str, value["ref"]),
        policy_key=cast(str, value["policy_key"]),
        identity_digest=cast(str, value["identity_digest"]),
        effect_identity=cast(str, value["effect_identity"]),
        durability=cast(Any, value["durability"]),
        recovery=cast(Any, value["recovery"]),
    )


def _context_sha256(context: Mapping[str, object]) -> str:
    """Bind omitted authored context without retaining it in source.db."""

    encoded = json.dumps(context, sort_keys=True, default=str, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _replay_plan_payload(plan: MutationPlan) -> dict[str, object]:
    """Persist only the plan fields the audit replay path consumes."""

    return {
        "operation": plan.operation,
        "destructive_class": plan.destructive_class,
        "target_refs": list(plan.target_refs),
        "affected_tiers": list(plan.affected_tiers),
        "reversible": plan.reversible,
        "prepared_at": plan.prepared_at,
        "plan_hash": plan.plan_hash,
        "context_sha256": _context_sha256(plan.context),
        "operation_version": plan.operation_version,
        "archive_instance_id": plan.archive_instance_id,
        "archive_identity_digest": plan.archive_identity_digest,
        "required_capabilities": list(plan.required_capabilities),
        "required_confirmation": plan.required_confirmation,
        "targets": [target.canonical_dict() for target in plan.targets],
        "parameter_digest": plan.parameter_digest,
        "target_digest": plan.target_digest,
        "prepared_at_ms": plan.prepared_at_ms,
        "expires_at_ms": plan.expires_at_ms,
    }


def _plan_from_payload(raw: object) -> MutationPlan:
    value = cast(dict[str, object], raw)
    raw_context = value.get("context")
    if raw_context is None:
        context_digest = value.get("context_sha256")
        if not isinstance(context_digest, str) or len(context_digest) != 64:
            raise ValueError("replayed plan lacks an authored-context digest")
        context: Mapping[str, object] = {}
    elif isinstance(raw_context, Mapping):
        # Compatibility for pending commands written before this replay-only
        # format. New commands never write authored context to source.db.
        context = cast(Mapping[str, object], raw_context)
    else:
        raise ValueError("replayed plan context is malformed")
    return MutationPlan(
        operation=cast(str, value["operation"]),
        destructive_class=cast(Any, value["destructive_class"]),
        target_refs=tuple(cast(list[str], value["target_refs"])),
        affected_tiers=tuple(cast(list[str], value["affected_tiers"])),
        reversible=cast(bool, value["reversible"]),
        prepared_at=cast(str, value["prepared_at"]),
        plan_hash=cast(str, value["plan_hash"]),
        context=context,
        operation_version=cast(int, value["operation_version"]),
        archive_instance_id=cast(str, value["archive_instance_id"]),
        archive_identity_digest=cast(str, value["archive_identity_digest"]),
        required_capabilities=tuple(cast(list[str], value["required_capabilities"])),
        required_confirmation=cast(Any, value["required_confirmation"]),
        targets=tuple(_target_from_payload(item) for item in cast(list[object], value["targets"])),
        parameter_digest=cast(str, value["parameter_digest"]),
        target_digest=cast(str, value["target_digest"]),
        prepared_at_ms=cast(int, value["prepared_at_ms"]),
        expires_at_ms=cast(int, value["expires_at_ms"]),
    )


def _principal_payload(principal: MutationPrincipal) -> dict[str, object]:
    return {
        "actor_ref": principal.actor_ref,
        "capabilities": sorted(principal.capabilities),
        "surface": principal.surface,
        "role_label": principal.role_label,
    }


def _principal_from_payload(raw: object) -> MutationPrincipal:
    value = cast(dict[str, object], raw)
    return MutationPrincipal(
        cast(str, value["actor_ref"]),
        frozenset(cast(list[str], value["capabilities"])),
        cast(Any, value["surface"]),
        cast(str | None, value.get("role_label")),
    )


def _preview_payload(preview: MutationPreview) -> dict[str, object]:
    return {"preview_ref": preview.preview_ref, "plan": _replay_plan_payload(preview.plan)}


def _preview_from_payload(raw: object) -> MutationPreview:
    value = cast(dict[str, object], raw)
    return MutationPreview(preview_ref=cast(str, value["preview_ref"]), plan=_plan_from_payload(value["plan"]))


def _authorization_payload(authorization: MutationAuthorization) -> dict[str, object]:
    return {
        **authorization.to_dict(),
        "token_sha256": None if authorization.token is None else token_sha256(authorization.token),
    }


def _authorization_from_payload(raw: object) -> MutationAuthorization:
    value = cast(dict[str, object], raw)
    return MutationAuthorization(
        plan_hash=cast(str, value["plan_hash"]),
        actor=cast(str, value["actor"]),
        role=cast(str, value["role"]),
        capability=cast(str, value["capability"]),
        confirmation_strength=cast(Any, value["confirmation_strength"]),
        authorized_at=cast(str, value["authorized_at"]),
        preview_ref=cast(str | None, value.get("preview_ref")),
        authorization_id=cast(str | None, value.get("authorization_id")),
        token=None,
        expires_at_ms=cast(int | None, value.get("expires_at_ms")),
        capabilities=tuple(cast(list[str], value["capabilities"])),
        surface=cast(Any, value.get("surface")),
    )


def _stored_authorization_digest(raw: object) -> _StoredAuthorizationDigest:
    value = cast(dict[str, object], raw)
    digest = value.get("token_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("replayed bound authorization lacks a token digest")
    return _StoredAuthorizationDigest(digest)


def _json_primitive(value: object) -> object:
    """Project typed receipt values into finite, replayable JSON primitives."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("continuity receipt contains a non-finite float")
        return value
    if isinstance(value, Enum):
        return _json_primitive(value.value)
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("continuity receipt object keys must be strings")
            normalized[key] = _json_primitive(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_primitive(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_primitive(model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        # Dataclass receipt values can carry private derived caches (for
        # example AnnotationBatch's canonical byte payload). Persist only the
        # constructor fields that define the replayable public value.
        return _json_primitive({field.name: getattr(value, field.name) for field in fields(value) if field.init})
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"continuity receipt cannot encode {type(value).__qualname__}")


def _receipt_payload(receipt: MutationReceipt) -> dict[str, object]:
    """Persist only finalization data consumed by the audit state transition."""

    return {
        "operation": receipt.operation,
        "plan_hash": receipt.plan_hash,
        "status": receipt.status,
        "target_refs": list(receipt.target_refs),
        "affected_count": receipt.affected_count,
        "receipt_ref": receipt.receipt_ref,
        "applied_at": receipt.applied_at,
        "operation_id": receipt.operation_id,
    }


def _receipt_from_payload(raw: object) -> MutationReceipt:
    value = cast(dict[str, object], raw)
    return MutationReceipt(
        operation=cast(str, value["operation"]),
        plan_hash=cast(str, value["plan_hash"]),
        status=cast(Any, value["status"]),
        target_refs=tuple(cast(list[str], value["target_refs"])),
        affected_count=cast(int, value["affected_count"]),
        detail=cast(str | None, value.get("detail")),
        receipt_ref=cast(str | None, value.get("receipt_ref")),
        applied_at=cast(str, value["applied_at"]),
        domain_receipt=cast(dict[str, object], value.get("domain_receipt", {})),
        operation_id=cast(str | None, value.get("operation_id")),
    )


class AuditRepository:
    """Small synchronous repository whose methods make audit transactions explicit."""

    def __init__(
        self,
        path: Path,
        *,
        attempt_owner_id: str | None = None,
        before_machine_prepare: Callable[[], None] | None = None,
        on_commit: Callable[[], None] | None = None,
    ) -> None:
        self.path = path
        self._attempt_owner_id = attempt_owner_id
        self._continuity = AuditContinuityCoordinator(path.parent)
        self._coordinated_connection: sqlite3.Connection | None = None
        self._coordinated_mutation: AuditMutation | None = None
        self._machine_binding: tuple[MachineRequestBinding, str] | None = None
        self._machine_part: int | None = None
        self._machine_deadline_unix_ms: int | None = None
        self._before_machine_prepare = before_machine_prepare
        self._on_commit = on_commit

    @contextmanager
    def bind_machine_request(
        self,
        binding: MachineRequestBinding,
        *,
        transition: str,
        part: int | None = None,
        deadline_unix_ms: int | None = None,
    ) -> Iterator[None]:
        """Bind exactly one domain transition in its existing continuity transaction."""

        if transition not in {
            "create_preview",
            "issue_authorization",
            "consume_authorization_and_start",
            "cancel_preview",
            "create_preview_batch",
            "issue_authorization_batch",
            "cancel_preview_batch",
            "accept_execution_batch",
            "accept_ingest",
        }:
            raise ValueError("machine request must bind a declared audit authority transition")
        if self._machine_binding is not None:
            raise RuntimeError("machine request binding scopes cannot overlap")
        if part is not None and (transition != "consume_authorization_and_start" or not 0 <= part < 40):
            raise ValueError("machine part must name a bounded execution ordinal")
        self._machine_binding = (binding, transition)
        self._machine_part = part
        self._machine_deadline_unix_ms = deadline_unix_ms
        try:
            yield
        finally:
            self._machine_binding = None
            self._machine_part = None
            self._machine_deadline_unix_ms = None

    def machine_request(self, binding: MachineRequestBinding) -> dict[str, object] | None:
        """Recover the immutable domain reference and reject conflicting reuse."""

        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM machine_requests WHERE archive_identity = ? AND request_id = ?",
                (binding.archive_identity, binding.request_id),
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        if any(record[key] != value for key, value in binding.to_dict().items()):
            raise MachineRequestConflictError("request id is bound to another principal or intent")
        return record

    def machine_request_for_principal(
        self, archive_identity: str, request_id: str, principal_ref: str
    ) -> dict[str, object] | None:
        """Resolve a control reference only in the authenticated current archive."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM machine_requests WHERE archive_identity = ? AND request_id = ? AND principal_ref = ?",
                (archive_identity, request_id, principal_ref),
            ).fetchone()
        return None if row is None else dict(row)

    @contextmanager
    def settled_machine_read(self) -> Iterator[dict[str, int]]:
        with self._continuity.settled_read() as versions:
            yield versions

    def preview_for_principal(self, preview_ref: str, principal: MutationPrincipal) -> MutationPreview:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT plan_json, principal_actor_ref, principal_surface FROM operation_previews WHERE preview_id = ?",
                (preview_ref,),
            ).fetchone()
        if row is None or row[1] != principal.actor_ref or row[2] != principal.surface:
            raise AuthorizationMismatchError("preview does not belong to the authenticated principal")
        return MutationPreview(preview_ref=preview_ref, plan=_plan_from_payload(json.loads(row[0])))

    def authorization_for_principal(
        self, authorization_ref: str, principal: MutationPrincipal
    ) -> tuple[MutationPreview, MutationAuthorization]:
        """Resolve authenticated durable authority without reconstructing a bearer token."""

        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM operation_authorizations WHERE authorization_id = ?", (authorization_ref,)
            ).fetchone()
            if row is None or row["actor_ref"] != principal.actor_ref or row["surface"] != principal.surface:
                raise AuthorizationMismatchError("authorization does not belong to the authenticated principal")
            capabilities = tuple(
                str(item[0])
                for item in conn.execute(
                    "SELECT capability FROM operation_authorization_capabilities WHERE authorization_id = ? ORDER BY capability",
                    (authorization_ref,),
                )
            )
            if not set(capabilities).issubset(principal.capabilities):
                raise AuthorizationMismatchError("authenticated principal no longer has authorization capabilities")
            record = dict(row)
        preview = self.preview_for_principal(str(record["preview_id"]), principal)
        authorization = MutationAuthorization(
            plan_hash=preview.plan.plan_hash,
            actor=principal.actor_ref,
            role=str(record["role_label"] or ""),
            capability=capabilities[0] if capabilities else "",
            confirmation_strength=cast(Any, record["confirmation_strength"]),
            authorized_at=str(record["issued_at_ms"]),
            preview_ref=preview.preview_ref,
            authorization_id=authorization_ref,
            token=None,
            expires_at_ms=int(cast(int, record["expires_at_ms"])),
            capabilities=capabilities,
            surface=principal.surface,
        )
        return preview, authorization

    @staticmethod
    def _bind_machine_result(conn: sqlite3.Connection, mutation: AuditMutation, result: object) -> None:
        raw = mutation.payload.get("machine_request")
        if raw is None:
            return
        binding = MachineRequestBinding(**cast(dict[str, str], raw))
        if mutation.kind == "accept_ingest":
            manifest = FrozenSourceManifest.from_dict(mutation.payload["manifest"])
            conn.execute(
                """INSERT INTO machine_requests(
                    archive_identity, request_id, principal_ref, fingerprint, operation_name,
                    artifact_kind, artifact_ref, accepted_at_ms, accepted_deadline_unix_ms
                ) VALUES (?, ?, ?, ?, ?, 'source-generation', ?, ?, ?)""",
                (
                    *binding.to_dict().values(),
                    manifest.source_generation_id,
                    mutation.created_at_ms,
                    mutation.payload.get("accepted_deadline_unix_ms"),
                ),
            )
            return
        if "machine_part" in mutation.payload:
            changed = conn.execute(
                """UPDATE machine_request_parts SET operation_id = ?
                WHERE archive_identity = ? AND request_id = ? AND ordinal = ? AND operation_id IS NULL""",
                (result, binding.archive_identity, binding.request_id, mutation.payload["machine_part"]),
            ).rowcount
            if changed != 1:
                raise MachineRequestConflictError("machine execution part has already started or is missing")
            return
        if mutation.kind in {
            "create_preview_batch",
            "issue_authorization_batch",
            "cancel_preview_batch",
            "accept_execution_batch",
        }:
            AuditRepository._bind_machine_batch(conn, mutation, binding, result)
            return
        plan = cast(
            dict[str, object],
            mutation.payload.get("plan") or cast(dict[str, object], mutation.payload["preview"])["plan"],
        )
        principal = mutation.payload.get("principal") or mutation.payload.get("authorization")
        if not isinstance(principal, dict):
            if mutation.kind != "cancel_preview":
                raise RuntimeError("machine acceptance lacks authenticated authority")
        elif (principal.get("actor_ref") or principal.get("actor")) != binding.principal_ref:
            raise MachineRequestConflictError("machine binding principal does not own the domain transition")
        if plan["archive_identity_digest"] != binding.archive_identity:
            raise MachineRequestConflictError("machine binding archive differs from domain authority")
        kinds = {
            "create_preview": "preview",
            "issue_authorization": "authorization",
            "consume_authorization_and_start": "operation",
            "cancel_preview": "cancelled-preview",
        }
        artifact = result
        if mutation.kind == "cancel_preview":
            artifact = cast(dict[str, object], mutation.payload["preview"])["preview_ref"]
        if artifact is None:
            return  # Expired authorization has no accepted domain execution.
        if not isinstance(artifact, str) or mutation.kind not in kinds:
            raise RuntimeError("machine acceptance requires a typed domain reference")
        conn.execute(
            """INSERT INTO machine_requests(
                archive_identity, request_id, principal_ref, fingerprint, operation_name,
                artifact_kind, artifact_ref, accepted_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (*binding.to_dict().values(), kinds[mutation.kind], artifact, mutation.created_at_ms),
        )

    @staticmethod
    def _bind_machine_batch(
        conn: sqlite3.Connection, mutation: AuditMutation, binding: MachineRequestBinding, result: object
    ) -> None:
        refs = cast(list[str], result)
        if not refs or len(refs) > 40:
            raise ValueError("machine authority batch must contain between 1 and 40 parts")
        kind = {
            "create_preview_batch": "preview-batch",
            "issue_authorization_batch": "authorization-batch",
            "cancel_preview_batch": "cancelled-preview-batch",
            "accept_execution_batch": "execution-batch",
        }[mutation.kind]
        conn.execute(
            """INSERT INTO machine_requests(
                archive_identity, request_id, principal_ref, fingerprint, operation_name,
                artifact_kind, artifact_ref, accepted_at_ms, part_count, accepted_deadline_unix_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                *binding.to_dict().values(),
                kind,
                refs[0],
                mutation.created_at_ms,
                len(refs),
                mutation.payload.get("accepted_deadline_unix_ms"),
            ),
        )
        for ordinal, ref in enumerate(refs):
            if kind in {"preview-batch", "cancelled-preview-batch"}:
                preview_ref, authorization_ref = ref, None
            else:
                row = conn.execute(
                    "SELECT preview_id FROM operation_authorizations WHERE authorization_id = ?", (ref,)
                ).fetchone()
                if row is None:
                    raise AuthorizationMismatchError("batch authorization is missing")
                preview_ref = str(row[0])
                # Authorization creation does not reserve execution. Only an
                # accepted execution owns the one-shot authority permanently.
                authorization_ref = ref if kind == "execution-batch" else None
            row = conn.execute(
                "SELECT principal_actor_ref, archive_identity_digest FROM operation_previews WHERE preview_id = ?",
                (preview_ref,),
            ).fetchone()
            if row is None or row[0] != binding.principal_ref or row[1] != binding.archive_identity:
                raise MachineRequestConflictError("batch authority differs from authenticated archive intent")
            conn.execute(
                """INSERT INTO machine_request_parts(
                    archive_identity, request_id, ordinal, artifact_ref, preview_ref, authorization_ref
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (binding.archive_identity, binding.request_id, ordinal, ref, preview_ref, authorization_ref),
            )

    def machine_parts(self, binding: MachineRequestBinding) -> list[dict[str, object]]:
        if self.machine_request(binding) is None:
            return []
        with self._connection() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM machine_request_parts WHERE archive_identity = ? AND request_id = ? ORDER BY ordinal",
                    (binding.archive_identity, binding.request_id),
                )
            ]

    def machine_preview_summary(self, binding: MachineRequestBinding) -> dict[str, object]:
        """Reconstruct a preview response from ordered normalized authority rows."""
        parts = self.machine_parts(binding)
        ids: list[str] = []
        expiries: list[int] = []
        refs = [str(part["preview_ref"]) for part in parts]
        with self._connection() as conn:
            for ref in refs:
                row = conn.execute(
                    "SELECT expires_at_ms FROM operation_previews WHERE preview_id = ?", (ref,)
                ).fetchone()
                if row is None:
                    raise ValueError("machine preview authority is missing")
                expiries.append(int(row[0]))
                ids.extend(
                    str(row[0]).removeprefix("session:")
                    for row in conn.execute(
                        "SELECT target_ref FROM operation_preview_targets WHERE preview_id = ? ORDER BY ordinal",
                        (ref,),
                    )
                )
        return {
            "status": "prepared",
            "operation": "delete",
            "preview_ref": refs[0],
            "preview_refs": refs,
            "session_ids": ids,
            "session_count": len(ids),
            "expires_at_ms": min(expiries),
        }

    @classmethod
    def for_archive_root(cls, archive_root: Path, *, attempt_owner_id: str | None = None) -> AuditRepository:
        """Build the repository for an already-initialized archive root."""

        return cls(archive_root / "audit.db", attempt_owner_id=attempt_owner_id)

    @staticmethod
    def current_process_attempt_owner() -> str:
        """Return the process identity assigned to production mutation attempts."""

        return _current_process_attempt_owner()

    def reconcile_continuity(self) -> None:
        """Reject audit bytes that cannot prove the source control head."""

        self._assert_regular_audit_leaf()
        self._continuity.reconcile(self._replay_pending_mutation)

    def _assert_regular_audit_leaf(self) -> None:
        """Refuse an audit pathname that redirects authority outside the archive root."""

        try:
            assert_verified_audit_leaf(self.path)
        except AuditLeafError as exc:
            raise RuntimeError(str(exc)) from exc

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self._assert_regular_audit_leaf()
        if self._coordinated_connection is not None:
            yield self._coordinated_connection
            return
        with open_verified_audit_connection(self.path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
            except BaseException:
                conn.rollback()
                raise
            else:
                conn.commit()

    def _continuity_payload(
        self, kind: str, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> dict[str, object]:
        """Encode exact typed replay inputs before source.db prepares a command."""

        values = dict(kwargs)
        if kind == "accept_ingest":
            manifest, principal = cast(FrozenSourceManifest, args[0]), cast(MutationPrincipal, args[1])
            if self._machine_binding is None or self._machine_binding[1] != kind:
                raise ValueError("ingest acceptance requires an authenticated machine binding")
            binding = self._machine_binding[0]
            if binding.principal_ref != principal.actor_ref or "archive.ingest" not in principal.capabilities:
                raise AuthorizationMismatchError("ingest acceptance principal lacks bound authority")
            return {"manifest": manifest.to_dict(), "principal": _principal_payload(principal)}
        if kind in {"create_preview_batch", "issue_authorization_batch", "cancel_preview_batch"}:
            items, principal = cast(tuple[object, ...], args[0]), cast(MutationPrincipal, args[1])
            if not 1 <= len(items) <= 40:
                raise ValueError("authority batch must contain between 1 and 40 parts")
            child_kind = {
                "create_preview_batch": "create_preview",
                "issue_authorization_batch": "issue_authorization",
                "cancel_preview_batch": "cancel_preview",
            }[kind]
            commands = []
            authorizations = cast(tuple[MutationAuthorization, ...], args[2]) if len(args) > 2 else ()
            if authorizations and len(authorizations) != len(items):
                raise ValueError("authorization batch differs from preview batch")
            for ordinal, item in enumerate(items):
                if child_kind == "cancel_preview":
                    child_args = (item,)
                elif child_kind == "create_preview":
                    child_args = (item, principal)
                else:
                    child_args = (item, principal, authorizations[ordinal])
                commands.append(
                    AuditMutation(
                        kind=child_kind,
                        mutation_id=f"audit-part:{secrets.token_urlsafe(18)}",
                        created_at_ms=int(time.time() * 1000),
                        payload=self._continuity_payload(child_kind, child_args, {}),
                    ).command()
                )
            return {"commands": commands}
        if kind == "accept_execution_batch":
            refs, principal = cast(tuple[str, ...], args[0]), cast(MutationPrincipal, args[1])
            if not 1 <= len(refs) <= 40 or len(set(refs)) != len(refs):
                raise ValueError("execution batch must contain between 1 and 40 distinct authorizations")
            return {
                "authorization_refs": list(refs),
                "principal": _principal_payload(principal),
                "now_ms": int(time.time() * 1000),
            }
        if kind == "stop_machine_batch":
            reason = str(args[1])
            if reason not in {"cancelled", "deadline", "refused", "indeterminate"}:
                raise ValueError("unknown machine stop reason")
            return {
                "binding": cast(MachineRequestBinding, args[0]).to_dict(),
                "reason": reason,
                "now_ms": int(time.time() * 1000),
            }
        if kind == "ensure_archive_authority":
            archive_instance_id = cast(str | None, values.get("archive_instance_id"))
            return {
                "now_ms": cast(int, values["now_ms"]),
                # Keep caller intent separate from the deterministic value used
                # only if this command has to create the authority row. A live
                # call with ``None`` accepts an existing id; replay must retain
                # that same optional semantic rather than treating a generated
                # value as an asserted authority id.
                "archive_instance_id": archive_instance_id,
                "generated_archive_instance_id": (
                    None if archive_instance_id is not None else f"archive:{secrets.token_hex(16)}"
                ),
            }
        if kind == "create_preview":
            plan, principal = cast(MutationPlan, args[0]), cast(MutationPrincipal, args[1])
            return {
                "preview_id": f"preview:{secrets.token_urlsafe(18)}",
                "plan": _replay_plan_payload(plan),
                "principal": _principal_payload(principal),
            }
        if kind == "issue_authorization":
            if isinstance(args[0], _StoredAuthorizationDigest):
                preview, principal, authorization = (
                    cast(MutationPreview, args[1]),
                    cast(MutationPrincipal, args[2]),
                    cast(MutationAuthorization, args[3]),
                )
            else:
                preview, principal, authorization = (
                    cast(MutationPreview, args[0]),
                    cast(MutationPrincipal, args[1]),
                    cast(MutationAuthorization, args[2]),
                )
            return {
                "authorization_id": f"authorization:{secrets.token_urlsafe(18)}",
                "issued_at_ms": cast(int, values.get("issued_at_ms", int(time.time() * 1000))),
                "preview": _preview_payload(preview),
                "principal": _principal_payload(principal),
                "authorization": _authorization_payload(authorization),
            }
        if kind == "consume_authorization_and_start":
            if isinstance(args[0], _StoredAuthorizationDigest):
                preview, authorization = cast(MutationPreview, args[1]), cast(MutationAuthorization, args[2])
            else:
                preview, authorization = cast(MutationPreview, args[0]), cast(MutationAuthorization, args[1])
            return {
                "operation_id": f"operation:{secrets.token_urlsafe(18)}",
                "attempt_id": f"attempt:{secrets.token_urlsafe(18)}",
                # The command can be replayed by a fresh repository process.
                # Keep the original actuator owner, rather than accidentally
                # assigning its pre-effect attempt to the recovery process.
                "attempt_owner_id": self._attempt_owner_id,
                "now_ms": int(time.time() * 1000),
                "preview": _preview_payload(preview),
                "authorization": {
                    **_authorization_payload(authorization),
                    "token_sha256": args[0].value
                    if isinstance(args[0], _StoredAuthorizationDigest)
                    else token_sha256(cast(str, authorization.token)),
                },
            }
        if kind == "mark_preview_stale":
            return {"preview": _preview_payload(cast(MutationPreview, args[0]))}
        if kind == "cancel_preview":
            return {"preview": _preview_payload(cast(MutationPreview, args[0]))}
        if kind == "finalize_attempt":
            operation_id = cast(str, args[0])
            return {
                "operation_id": operation_id,
                "status": cast(str, values["status"]),
                "receipt": None
                if values.get("receipt") is None
                else _receipt_payload(cast(MutationReceipt, values["receipt"])),
                "error_summary": values.get("error_summary"),
                "unknown_reason": values.get("unknown_reason"),
                "now_ms": int(time.time() * 1000),
            }
        if kind == "recover_abandoned_attempts":
            return {"now_ms": int(time.time() * 1000)}
        if kind == "record_recovery_disposition":
            operation_id = cast(str, args[0])
            disposition = cast(RecoveryDisposition, args[1])
            return {
                "operation_id": operation_id,
                "kind": disposition.kind,
                "action": disposition.action,
                "detail": disposition.detail,
                "target_dispositions": [
                    {
                        "target_ref": item.target_ref,
                        "state": item.state,
                        "action": item.action,
                        "detail": item.detail,
                    }
                    for item in disposition.target_dispositions
                ],
                "evidence_ref": disposition.evidence_ref,
                "now_ms": int(time.time() * 1000),
            }
        if kind == "adjudicate_recovery":
            return {
                "operation_id": cast(str, args[0]),
                "target_outcomes": cast(Mapping[str, str], values["target_outcomes"]),
                "reason": cast(str, values["reason"]),
                "now_ms": int(time.time() * 1000),
            }
        raise RuntimeError(f"unregistered audit continuity mutation {kind!r}")

    def _replay_pending_mutation(self, conn: sqlite3.Connection, mutation: AuditMutation) -> object:
        result = self._replay_domain_mutation(conn, mutation)
        self._bind_machine_result(conn, mutation, result)
        return result

    def _replay_domain_mutation(self, conn: sqlite3.Connection, mutation: AuditMutation) -> object:
        """Replay the stored typed command without allocating fresh ids or clocks."""

        payload = mutation.payload
        self._coordinated_connection = conn
        self._coordinated_mutation = mutation
        try:
            if mutation.kind in {"create_preview_batch", "issue_authorization_batch", "cancel_preview_batch"}:
                return self._apply_authority_batch()
            if mutation.kind == "accept_ingest":
                # The source-WAL prepare has already accepted this denominator.
                # Replaying the audit reference cannot reauthorize or acquire.
                return FrozenSourceManifest.from_dict(payload["manifest"]).source_generation_id
            if mutation.kind == "accept_execution_batch":
                return cast(Any, self.accept_execution_batch).__wrapped__(
                    self,
                    tuple(cast(list[str], payload["authorization_refs"])),
                    _principal_from_payload(payload["principal"]),
                )
            if mutation.kind == "stop_machine_batch":
                return cast(Any, self.stop_machine_batch).__wrapped__(
                    self, MachineRequestBinding(**cast(dict[str, str], payload["binding"])), str(payload["reason"])
                )
            if mutation.kind == "ensure_archive_authority":
                return cast(Any, self.ensure_archive_authority).__wrapped__(
                    self,
                    now_ms=cast(int, payload["now_ms"]),
                    archive_instance_id=cast(str | None, payload.get("archive_instance_id")),
                )
            if mutation.kind == "create_preview":
                return cast(Any, self.create_preview).__wrapped__(
                    self, _plan_from_payload(payload["plan"]), _principal_from_payload(payload["principal"])
                )
            if mutation.kind == "issue_authorization":
                return self._persist_authorization(
                    _stored_authorization_digest(payload["authorization"]),
                    _preview_from_payload(payload["preview"]),
                    _principal_from_payload(payload["principal"]),
                    _authorization_from_payload(payload["authorization"]),
                    issued_at_ms=cast(int, payload["issued_at_ms"]),
                )
            if mutation.kind == "consume_authorization_and_start":
                return self._consume_authorization(
                    _stored_authorization_digest(payload["authorization"]),
                    _preview_from_payload(payload["preview"]),
                    _authorization_from_payload(payload["authorization"]),
                )
            if mutation.kind == "mark_preview_stale":
                return cast(Any, self.mark_preview_stale).__wrapped__(
                    self,
                    _preview_from_payload(payload["preview"]),
                )
            if mutation.kind == "cancel_preview":
                return cast(Any, self.cancel_preview).__wrapped__(
                    self,
                    _preview_from_payload(payload["preview"]),
                )
            if mutation.kind == "finalize_attempt":
                return cast(Any, self.finalize_attempt).__wrapped__(
                    self,
                    cast(str, payload["operation_id"]),
                    status=cast(str, payload["status"]),
                    receipt=None if payload["receipt"] is None else _receipt_from_payload(payload["receipt"]),
                    error_summary=cast(str | None, payload.get("error_summary")),
                    unknown_reason=cast(str | None, payload.get("unknown_reason")),
                )
            if mutation.kind == "recover_abandoned_attempts":
                return cast(Any, self._recover_abandoned_attempts).__wrapped__(self)
            if mutation.kind == "record_recovery_disposition":
                return cast(Any, self.record_recovery_disposition).__wrapped__(
                    self,
                    cast(str, payload["operation_id"]),
                    RecoveryDisposition(
                        kind=cast(Any, payload["kind"]),
                        action=cast(Any, payload["action"]),
                        detail=cast(str | None, payload.get("detail")),
                        target_dispositions=tuple(
                            RecoveryTargetDisposition(
                                target_ref=cast(str, item["target_ref"]),
                                state=cast(Any, item["state"]),
                                action=cast(Any, item["action"]),
                                detail=cast(str | None, item.get("detail")),
                            )
                            for item in cast(list[dict[str, object]], payload.get("target_dispositions", []))
                        ),
                        evidence_ref=cast(str | None, payload.get("evidence_ref")),
                    ),
                )
            if mutation.kind == "adjudicate_recovery":
                return cast(Any, self.adjudicate_recovery).__wrapped__(
                    self,
                    cast(str, payload["operation_id"]),
                    target_outcomes=cast(
                        Mapping[str, Literal["applied", "not-applied", "unknown"]], payload["target_outcomes"]
                    ),
                    reason=cast(str, payload["reason"]),
                )
            raise RuntimeError(f"unregistered audit continuity mutation {mutation.kind!r}")
        finally:
            self._coordinated_mutation = None
            self._coordinated_connection = None

    def _command_value(self, key: str, fallback: object) -> object:
        if self._coordinated_mutation is None:
            return fallback
        return self._coordinated_mutation.payload.get(key, fallback)

    def _begin(self, conn: sqlite3.Connection) -> None:
        """Start a standalone audit transaction, or reuse the coordinator's one."""

        if self._coordinated_connection is None:
            conn.execute("BEGIN IMMEDIATE")

    def _apply_authority_batch(self) -> list[str]:
        outer, conn = self._coordinated_mutation, self._coordinated_connection
        if outer is None or conn is None:
            raise RuntimeError("authority batches require source-WAL continuity")
        results: list[str] = []
        try:
            for command in cast(list[dict[str, object]], outer.payload["commands"]):
                result = self._replay_domain_mutation(conn, AuditMutation.from_command(command))
                if outer.kind == "cancel_preview_batch":
                    result = cast(dict[str, object], cast(dict[str, object], command["payload"])["preview"])[
                        "preview_ref"
                    ]
                if not isinstance(result, str):
                    raise TokenExpiredError("authority batch includes an expired preview")
                results.append(result)
            return results
        finally:
            self._coordinated_mutation, self._coordinated_connection = outer, conn

    @_continuity_mutation("create_preview_batch")
    def create_preview_batch(self, plans: tuple[MutationPlan, ...], principal: MutationPrincipal) -> list[str]:
        """Publish a bounded ordered preview set in one continuity transition."""
        return self._apply_authority_batch()

    @_continuity_mutation("issue_authorization_batch")
    def issue_authorization_batch(
        self,
        previews: tuple[MutationPreview, ...],
        principal: MutationPrincipal,
        authorizations: tuple[MutationAuthorization, ...],
    ) -> list[str]:
        """Publish exact one-shot references without retaining bearer tokens."""
        return self._apply_authority_batch()

    @_continuity_mutation("accept_ingest")
    def accept_ingest(self, manifest: FrozenSourceManifest, principal: MutationPrincipal) -> str:
        """Bind retained physical inputs; source preparation owns acceptance."""
        return manifest.source_generation_id

    @_continuity_mutation("accept_execution_batch")
    def accept_execution_batch(self, refs: tuple[str, ...], principal: MutationPrincipal) -> list[str]:
        """Reserve ordered authority; a domain run is created only before its effect."""
        now_ms = int(cast(int, self._command_value("now_ms", int(time.time() * 1000))))
        with self._connection() as conn:
            for ref in refs:
                row = conn.execute(
                    """SELECT a.actor_ref, a.surface, a.state, a.expires_at_ms, p.state
                    FROM operation_authorizations a JOIN operation_previews p ON p.preview_id = a.preview_id
                    WHERE a.authorization_id = ?""",
                    (ref,),
                ).fetchone()
                if row is None or row[0] != principal.actor_ref or row[1] != principal.surface:
                    raise AuthorizationMismatchError("execution authority belongs to another principal")
                if row[2] != "active" or row[4] != "prepared":
                    raise TokenConsumedError("execution authority is no longer active")
                if int(row[3]) <= now_ms:
                    raise TokenExpiredError("execution authority is expired")
                capabilities = {
                    str(r[0])
                    for r in conn.execute(
                        "SELECT capability FROM operation_authorization_capabilities WHERE authorization_id = ?", (ref,)
                    )
                }
                if not capabilities.issubset(principal.capabilities):
                    raise AuthorizationMismatchError("execution principal lacks reserved capabilities")
                if conn.execute("SELECT 1 FROM machine_request_parts WHERE authorization_ref = ?", (ref,)).fetchone():
                    raise TokenConsumedError("execution authority is already reserved")
        return list(refs)

    @_continuity_mutation("cancel_preview_batch")
    def cancel_preview_batch(self, previews: tuple[MutationPreview, ...], principal: MutationPrincipal) -> list[str]:
        """Cancel an exact ordered preview set without consuming its authority."""
        return self._apply_authority_batch()

    @_continuity_mutation("stop_machine_batch")
    def stop_machine_batch(self, binding: MachineRequestBinding, reason: str) -> None:
        """Fence unstarted parts without hiding or revoking an attempted effect."""
        if self.machine_request(binding) is None:
            raise ValueError("machine request is unknown")
        with self._connection() as conn:
            self._begin(conn)
            conn.execute(
                """UPDATE machine_requests SET stop_reason = COALESCE(stop_reason, ?),
                    stopped_at_ms = COALESCE(stopped_at_ms, ?)
                WHERE archive_identity = ? AND request_id = ?""",
                (
                    reason,
                    self._command_value("now_ms", int(time.time() * 1000)),
                    binding.archive_identity,
                    binding.request_id,
                ),
            )
            conn.execute(
                """UPDATE operation_authorizations SET state = 'revoked'
                WHERE state = 'active' AND authorization_id IN (
                    SELECT authorization_ref FROM machine_request_parts
                    WHERE archive_identity = ? AND request_id = ? AND operation_id IS NULL
                )""",
                (binding.archive_identity, binding.request_id),
            )

    @_continuity_mutation("ensure_archive_authority")
    def ensure_archive_authority(self, *, now_ms: int, archive_instance_id: str | None = None) -> str:
        """Create or return the immutable archive lineage id."""

        with self._connection() as conn:
            row = conn.execute("SELECT archive_instance_id FROM archive_authority LIMIT 1").fetchone()
            if row is not None:
                existing = str(row[0])
                if archive_instance_id is not None and archive_instance_id != existing:
                    raise ValueError("audit archive instance identity changed")
                return existing
            instance_id = cast(
                str,
                archive_instance_id
                or self._command_value(
                    "generated_archive_instance_id",
                    self._command_value("archive_instance_id", ""),
                ),
            )
            if not instance_id:
                raise RuntimeError("audit archive authority command lacks an instance identity")
            conn.execute(
                "INSERT INTO archive_authority(archive_instance_id, created_at_ms, authority_format) VALUES (?, ?, 1)",
                (instance_id, now_ms),
            )
            return instance_id

    @_continuity_mutation("create_preview")
    def create_preview(self, plan: MutationPlan, principal: MutationPrincipal) -> str:
        """Persist a bounded preview and its normalized target/capability rows."""

        preview_id = cast(str, self._command_value("preview_id", f"preview:{secrets.token_urlsafe(18)}"))
        with self._connection() as conn:
            self._begin(conn)
            conn.execute(
                """
                INSERT INTO operation_previews (
                    preview_id, operation_name, operation_version, archive_instance_id,
                    archive_identity_digest, plan_hash, parameter_digest, target_digest,
                    target_count, destructive_class, required_confirmation,
                    required_capability_count, principal_actor_ref, principal_surface,
                    role_label, state, created_at_ms, expires_at_ms, plan_format, plan_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?,
                          'polylogue.mutation-plan/v1', ?)
                """,
                (
                    preview_id,
                    plan.operation,
                    plan.operation_version,
                    plan.archive_instance_id,
                    plan.archive_identity_digest,
                    plan.plan_hash,
                    plan.parameter_digest,
                    plan.target_digest or plan.plan_hash,
                    plan.target_count,
                    plan.destructive_class,
                    plan.required_confirmation,
                    len(plan.required_capabilities),
                    principal.actor_ref,
                    principal.surface,
                    principal.role_label,
                    plan.prepared_at_ms,
                    plan.expires_at_ms,
                    json.dumps(
                        {
                            **_replay_plan_payload(plan),
                            "context": dict(plan.context),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            for ordinal, target in enumerate(plan.targets):
                conn.execute(
                    """
                    INSERT INTO operation_preview_targets(
                        preview_id, ordinal, target_kind, target_ref, identity_digest,
                        effect_identity, durability, recovery_policy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        preview_id,
                        ordinal,
                        target.kind,
                        target.ref,
                        target.identity_digest,
                        target.effect_identity,
                        target.durability,
                        target.recovery,
                    ),
                )
            for capability in plan.required_capabilities:
                conn.execute(
                    "INSERT INTO operation_preview_capabilities(preview_id, capability) VALUES (?, ?)",
                    (preview_id, capability),
                )
        return preview_id

    def issue_authorization(
        self,
        preview: MutationPreview,
        principal: MutationPrincipal,
        authorization: MutationAuthorization,
        *,
        issued_at_ms: int | None = None,
    ) -> str:
        """Persist token digest and exact proved capabilities, never token material."""

        if authorization.token is None:
            raise ValueError("bound authorization requires a token")
        authorization_id = self._issue_authorization(
            _StoredAuthorizationDigest(token_sha256(authorization.token)),
            preview,
            principal,
            authorization,
            issued_at_ms=issued_at_ms,
        )
        if authorization_id is None:
            raise TokenExpiredError("cannot authorize an expired preview")
        return authorization_id

    @_continuity_mutation("issue_authorization")
    def _issue_authorization(
        self,
        token_digest: _StoredAuthorizationDigest,
        preview: MutationPreview,
        principal: MutationPrincipal,
        authorization: MutationAuthorization,
        *,
        issued_at_ms: int | None = None,
    ) -> str | None:
        """Persist one token from the durable preview's exact authorization evidence."""

        return self._persist_authorization(
            token_digest,
            preview,
            principal,
            authorization,
            issued_at_ms=issued_at_ms,
        )

    def _persist_authorization(
        self,
        token_digest: _StoredAuthorizationDigest,
        preview: MutationPreview,
        principal: MutationPrincipal,
        authorization: MutationAuthorization,
        *,
        issued_at_ms: int | None = None,
    ) -> str | None:
        authorization_id = cast(
            str, self._command_value("authorization_id", f"authorization:{secrets.token_urlsafe(18)}")
        )
        effective_issued_at_ms = cast(
            int,
            self._command_value("issued_at_ms", issued_at_ms if issued_at_ms is not None else int(time.time() * 1000)),
        )
        with self._connection() as conn:
            self._begin(conn)
            preview_row = conn.execute(
                """
                SELECT plan_hash, expires_at_ms, state, principal_actor_ref,
                       principal_surface, role_label, required_confirmation
                FROM operation_previews WHERE preview_id = ?
                """,
                (preview.preview_ref,),
            ).fetchone()
            if preview_row is None:
                raise ValueError(f"unknown preview {preview.preview_ref!r}")
            if str(preview_row[0]) != preview.plan.plan_hash:
                raise ValueError("preview plan hash does not match its durable row")
            if str(preview_row[2]) != "prepared":
                raise ValueError("preview is not authorizable")
            durable_expires_at_ms = int(preview_row[1])
            durable_capabilities = tuple(
                str(row[0])
                for row in conn.execute(
                    "SELECT capability FROM operation_preview_capabilities WHERE preview_id = ? ORDER BY capability",
                    (preview.preview_ref,),
                )
            )
            if principal.actor_ref != str(preview_row[3]) or principal.surface != str(preview_row[4]):
                raise ValueError("authorization principal differs from preview principal")
            if principal.role_label != cast(str | None, preview_row[5]):
                raise ValueError("authorization role differs from preview principal")
            if not set(durable_capabilities).issubset(principal.capabilities):
                raise ValueError("authorization principal lacks the preview's required capabilities")
            if (
                _CONFIRMATION_STRENGTH_ORDER.get(authorization.confirmation_strength, -1)
                < _CONFIRMATION_STRENGTH_ORDER[str(preview_row[6])]
            ):
                raise ValueError("authorization confirmation is weaker than the durable preview")
            if (
                authorization.preview_ref != preview.preview_ref
                or authorization.plan_hash != str(preview_row[0])
                or authorization.actor != str(preview_row[3])
                or authorization.surface != str(preview_row[4])
                or authorization.role != (cast(str | None, preview_row[5]) or "")
                or authorization.expires_at_ms != durable_expires_at_ms
                or authorization.capabilities != durable_capabilities
                or (durable_capabilities and authorization.capability not in durable_capabilities)
                or (not durable_capabilities and authorization.capability != "")
            ):
                raise ValueError("authorization evidence differs from its durable preview")
            if effective_issued_at_ms >= durable_expires_at_ms:
                conn.execute(
                    "UPDATE operation_previews SET state = 'expired' WHERE preview_id = ? AND state = 'prepared'",
                    (preview.preview_ref,),
                )
                return None
            conn.execute(
                """
                UPDATE operation_authorizations
                SET state = 'revoked'
                WHERE preview_id = ? AND state = 'active'
                """,
                (preview.preview_ref,),
            )
            conn.execute(
                """
                INSERT INTO operation_authorizations(
                    authorization_id, preview_id, actor_ref, surface, role_label,
                    confirmation_strength, token_sha256, state, issued_at_ms,
                    expires_at_ms, consumed_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, NULL)
                """,
                (
                    authorization_id,
                    preview.preview_ref,
                    principal.actor_ref,
                    principal.surface,
                    principal.role_label,
                    authorization.confirmation_strength,
                    token_digest.value,
                    effective_issued_at_ms,
                    durable_expires_at_ms,
                ),
            )
            for capability in durable_capabilities:
                conn.execute(
                    "INSERT INTO operation_authorization_capabilities(authorization_id, capability) VALUES (?, ?)",
                    (authorization_id, capability),
                )
        return authorization_id

    @_continuity_mutation("mark_preview_stale")
    def mark_preview_stale(self, preview: MutationPreview) -> None:
        """Revoke every live authorization when a prepared plan no longer matches."""

        with self._connection() as conn:
            self._begin(conn)
            row = conn.execute(
                "SELECT plan_hash FROM operation_previews WHERE preview_id = ?",
                (preview.preview_ref,),
            ).fetchone()
            if row is None or str(row[0]) != preview.plan.plan_hash:
                raise ValueError("stale preview does not match its durable authority")
            conn.execute(
                """
                UPDATE operation_authorizations
                SET state = 'revoked'
                WHERE preview_id = ? AND state = 'active'
                """,
                (preview.preview_ref,),
            )
            conn.execute(
                "UPDATE operation_previews SET state = 'stale' WHERE preview_id = ? AND state = 'prepared'",
                (preview.preview_ref,),
            )

    @_continuity_mutation("cancel_preview")
    def cancel_preview(self, preview: MutationPreview) -> None:
        """Cancel an unconfirmed preview and revoke any live authorization."""

        with self._connection() as conn:
            self._begin(conn)
            row = conn.execute(
                "SELECT plan_hash FROM operation_previews WHERE preview_id = ?",
                (preview.preview_ref,),
            ).fetchone()
            if row is None or str(row[0]) != preview.plan.plan_hash:
                raise ValueError("cancelled preview does not match its durable authority")
            conn.execute(
                """
                UPDATE operation_authorizations
                SET state = 'revoked'
                WHERE preview_id = ? AND state = 'active'
                """,
                (preview.preview_ref,),
            )
            conn.execute(
                "UPDATE operation_previews SET state = 'cancelled' WHERE preview_id = ? AND state = 'prepared'",
                (preview.preview_ref,),
            )

    def consume_authorization_and_start(self, preview: MutationPreview, authorization: MutationAuthorization) -> str:
        """Consume a token and create run, targets, and initial attempt atomically."""

        if authorization.token is None:
            if authorization.authorization_id is None:
                raise ValueError("authorization token or durable reference is missing")
            with self._connection() as conn:
                row = conn.execute(
                    "SELECT token_sha256 FROM operation_authorizations WHERE authorization_id = ?",
                    (authorization.authorization_id,),
                ).fetchone()
            if row is None:
                raise AuthorizationMismatchError("authorization reference is unknown")
            digest = str(row[0])
        else:
            digest = token_sha256(authorization.token)
        operation_id = self._consume_authorization_and_start(
            _StoredAuthorizationDigest(digest),
            preview,
            authorization,
        )
        if operation_id is None:
            raise TokenExpiredError("authorization token is expired")
        return operation_id

    @_continuity_mutation("consume_authorization_and_start")
    def _consume_authorization_and_start(
        self,
        token_digest: _StoredAuthorizationDigest,
        preview: MutationPreview,
        authorization: MutationAuthorization,
    ) -> str | None:
        """Commit an expired-token transition before reporting it to the caller."""

        return self._consume_authorization(token_digest, preview, authorization)

    def _consume_authorization(
        self,
        token_digest: _StoredAuthorizationDigest,
        preview: MutationPreview,
        authorization: MutationAuthorization,
    ) -> str | None:
        validate_mutation_plan_integrity(preview.plan)
        operation_id = cast(str, self._command_value("operation_id", f"operation:{secrets.token_urlsafe(18)}"))
        attempt_id = cast(str, self._command_value("attempt_id", f"attempt:{secrets.token_urlsafe(18)}"))
        now_ms = cast(int, self._command_value("now_ms", int(time.time() * 1000)))
        with self._connection() as conn:
            self._begin(conn)
            row = conn.execute(
                """
                SELECT a.authorization_id, a.preview_id, a.actor_ref, a.surface,
                       a.role_label, a.confirmation_strength, a.state, a.expires_at_ms,
                       p.plan_hash
                FROM operation_authorizations AS a
                JOIN operation_previews AS p ON p.preview_id = a.preview_id
                WHERE a.token_sha256 = ?
                """,
                (token_digest.value,),
            ).fetchone()
            if row is None or str(row[1]) != preview.preview_ref:
                raise AuthorizationMismatchError("authorization token does not match preview")
            reservation = conn.execute(
                """SELECT p.archive_identity, p.request_id, p.ordinal, p.operation_id, r.stop_reason
                FROM machine_request_parts p JOIN machine_requests r
                    ON r.archive_identity = p.archive_identity AND r.request_id = p.request_id
                WHERE p.authorization_ref = ?""",
                (str(row[0]),),
            ).fetchone()
            command = {} if self._coordinated_mutation is None else self._coordinated_mutation.payload
            if reservation is None and "machine_part" in command:
                raise TokenConsumedError("authorization is not reserved to this machine part")
            if reservation is not None:
                bound = command.get("machine_request")
                if (
                    not isinstance(bound, dict)
                    or (bound.get("archive_identity"), bound.get("request_id"), command.get("machine_part"))
                    != (reservation[0], reservation[1], reservation[2])
                    or reservation[3] is not None
                    or reservation[4] is not None
                ):
                    raise TokenConsumedError("authorization is reserved to an accepted machine request")
            if str(row[6]) != "active":
                raise TokenConsumedError("authorization token is already consumed or revoked")
            if int(row[7]) <= now_ms:
                conn.execute(
                    "UPDATE operation_authorizations SET state = 'expired' WHERE authorization_id = ?",
                    (str(row[0]),),
                )
                return None
            durable_capabilities = tuple(
                str(capability_row[0])
                for capability_row in conn.execute(
                    "SELECT capability FROM operation_authorization_capabilities WHERE authorization_id = ? ORDER BY capability",
                    (str(row[0]),),
                )
            )
            if (
                str(row[2]) != authorization.actor
                or str(row[3]) != (authorization.surface or "")
                or (cast(str | None, row[4]) or "") != authorization.role
                or str(row[5]) != authorization.confirmation_strength
                or int(row[7]) != authorization.expires_at_ms
                or durable_capabilities != authorization.capabilities
                or (durable_capabilities and authorization.capability not in durable_capabilities)
                or (not durable_capabilities and authorization.capability != "")
            ):
                raise AuthorizationMismatchError("authorization principal mismatch")
            if str(row[8]) != preview.plan.plan_hash or authorization.plan_hash != preview.plan.plan_hash:
                raise AuthorizationMismatchError("authorization plan mismatch")
            conn.execute(
                "UPDATE operation_authorizations SET state = 'consumed', consumed_at_ms = ? WHERE authorization_id = ?",
                (now_ms, str(row[0])),
            )
            conn.execute(
                "UPDATE operation_previews SET state = 'consumed' WHERE preview_id = ?",
                (preview.preview_ref,),
            )
            conn.execute(
                """
                INSERT INTO operation_runs(
                    operation_id, preview_id, initial_authorization_id,
                    operation_name, operation_version, archive_instance_id,
                    archive_identity_digest, plan_hash, parameter_digest,
                    target_digest, target_count, actor_ref, surface, role_label,
                    status, requested_at_ms, started_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    operation_id,
                    preview.preview_ref,
                    str(row[0]),
                    preview.plan.operation,
                    preview.plan.operation_version,
                    preview.plan.archive_instance_id,
                    preview.plan.archive_identity_digest,
                    preview.plan.plan_hash,
                    preview.plan.parameter_digest,
                    preview.plan.target_digest or preview.plan.plan_hash,
                    preview.plan.target_count,
                    str(row[2]),
                    str(row[3]),
                    cast(str | None, row[4]),
                    now_ms,
                    now_ms,
                    now_ms,
                ),
            )
            for capability in durable_capabilities:
                conn.execute(
                    "INSERT INTO operation_run_capabilities(operation_id, capability) VALUES (?, ?)",
                    (operation_id, capability),
                )
            for ordinal, target in enumerate(preview.plan.targets):
                conn.execute(
                    """
                    INSERT INTO operation_targets(
                        operation_id, ordinal, target_kind, target_ref, identity_digest,
                        effect_identity, state, attempt_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        operation_id,
                        ordinal,
                        target.kind,
                        target.ref,
                        target.identity_digest,
                        target.effect_identity,
                        "running" if ordinal == 0 else "pending",
                    ),
                )
            conn.execute(
                """
                INSERT INTO operation_attempts(
                    attempt_id, operation_id, target_ordinal, authorization_id,
                    worker_id, state, started_at_ms
                ) VALUES (?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    attempt_id,
                    operation_id,
                    0 if preview.plan.targets else None,
                    str(row[0]),
                    cast(str | None, self._command_value("attempt_owner_id", self._attempt_owner_id)),
                    now_ms,
                ),
            )
            self._append_event(
                conn,
                operation_id=operation_id,
                event_type="authorization_consumed",
                to_state="running",
                actor_ref=authorization.actor,
                occurred_at_ms=now_ms,
                detail={"target_count": preview.plan.target_count},
            )
        return operation_id

    @_continuity_mutation("finalize_attempt")
    def finalize_attempt(
        self,
        operation_id: str,
        *,
        status: str,
        receipt: MutationReceipt | None = None,
        error_summary: str | None = None,
        unknown_reason: str | None = None,
    ) -> None:
        """Finalize one running attempt and parent run in one audit transaction."""

        now_ms = cast(int, self._command_value("now_ms", int(time.time() * 1000)))
        target_state = cast(
            AuditTargetState,
            {
                "unknown": "unknown",
                "failed": "failed",
                "blocked": "rejected",
                "already_satisfied": "already_satisfied",
            }.get(status, "applied"),
        )
        attempt_state = (
            "unknown"
            if target_state == "unknown"
            else "failed"
            if target_state in {"rejected", "failed"}
            else "applied"
        )
        with self._connection() as conn:
            self._begin(conn)
            run = conn.execute(
                """
                SELECT operation_name, plan_hash, target_digest, target_count, actor_ref
                FROM operation_runs WHERE operation_id = ?
                """,
                (operation_id,),
            ).fetchone()
            if run is None:
                raise ValueError(f"unknown operation {operation_id!r}")
            if receipt is not None:
                operation_name, plan_hash, _target_digest, target_count, _actor_ref = run
                durable_refs = tuple(
                    str(row[0])
                    for row in conn.execute(
                        "SELECT target_ref FROM operation_targets WHERE operation_id = ? ORDER BY ordinal",
                        (operation_id,),
                    )
                )
                if receipt.operation != str(operation_name):
                    raise ValueError("mutation receipt operation does not match the audited operation")
                if receipt.plan_hash != str(plan_hash):
                    raise ValueError("mutation receipt plan does not match the audited plan")
                if receipt.status != status:
                    raise ValueError("mutation receipt status does not match the audited outcome")
                if receipt.target_refs != durable_refs or len(durable_refs) != int(target_count):
                    raise ValueError("mutation receipt targets do not match the audited target set")
                if receipt.operation_id not in (None, operation_id):
                    raise ValueError("mutation receipt operation id does not match the audited operation")
                receipt = replace(
                    receipt,
                    receipt_ref=receipt.receipt_ref or f"mutation-operation:{operation_id}",
                    operation_id=operation_id,
                )
            target = conn.execute(
                "SELECT ordinal FROM operation_targets WHERE operation_id = ? AND state IN ('running', 'pending') ORDER BY ordinal LIMIT 1",
                (operation_id,),
            ).fetchone()
            ordinal = int(target[0]) if target is not None else None
            conn.execute(
                """
                UPDATE operation_attempts
                SET state = ?, finished_at_ms = ?, error_summary = ?, unknown_reason = ?
                WHERE operation_id = ? AND state = 'running'
                """,
                (attempt_state, now_ms, error_summary, unknown_reason, operation_id),
            )
            conn.execute(
                """
                UPDATE operation_targets
                SET state = ?, completed_at_ms = ?, error_summary = ?, unknown_reason = ?,
                    domain_receipt_ref = ?, domain_receipt_kind = ?
                WHERE operation_id = ? AND state IN ('running', 'pending')
                """,
                (
                    target_state,
                    now_ms,
                    error_summary,
                    unknown_reason,
                    None if receipt is None else receipt.receipt_ref,
                    None if receipt is None else "mutation-receipt",
                    operation_id,
                ),
            )
            states = [
                str(row[0])
                for row in conn.execute("SELECT state FROM operation_targets WHERE operation_id = ?", (operation_id,))
            ]
            run_status, terminal_reason = _run_state_for_targets(states)
            conn.execute(
                """
                UPDATE operation_runs
                SET status = ?, terminal_reason = ?, updated_at_ms = ?,
                    completed_at_ms = CASE WHEN ? IN ('completed', 'failed', 'interrupted') THEN ? ELSE completed_at_ms END,
                    rejected_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'rejected'),
                    failed_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'failed'),
                    unknown_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'unknown'),
                    affected_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'applied'),
                    error_summary = ?, unknown_reason = ?,
                    domain_receipt_ref = ?, domain_receipt_kind = ?
                WHERE operation_id = ?
                """,
                (
                    run_status,
                    terminal_reason,
                    now_ms,
                    run_status,
                    now_ms,
                    operation_id,
                    operation_id,
                    operation_id,
                    operation_id,
                    error_summary,
                    unknown_reason,
                    None if receipt is None else receipt.receipt_ref,
                    None if receipt is None else "mutation-receipt",
                    operation_id,
                ),
            )
            self._append_event(
                conn,
                operation_id=operation_id,
                target_ordinal=ordinal,
                event_type="attempt_finalized" if run_status != "interrupted" else "attempt_unknown",
                from_state="running",
                to_state=run_status,
                actor_ref=str(run[4]),
                occurred_at_ms=now_ms,
                detail=_receipt_event_detail(receipt, status=status, reason=unknown_reason or error_summary),
            )

    def recover_abandoned_attempts(self) -> tuple[str, ...]:
        """Recover only work that a prior process actually left running."""

        with self._connection() as conn:
            has_running = conn.execute("SELECT 1 FROM operation_attempts WHERE state = 'running' LIMIT 1").fetchone()
        if has_running is None:
            return ()
        return self._recover_abandoned_attempts()

    def nonterminal_operations_overlapping(self, target_refs: tuple[str, ...]) -> tuple[RecoveryOperation, ...]:
        """Return durable nonterminal operations touching an exact target set.

        This deliberately discovers both running and already-interrupted work.
        Callers must inspect the returned domain targets before issuing a new
        authorization.  Audit rows provide identity and plan binding only;
        they never stand in for target-state inspection.
        """

        if not target_refs:
            return ()
        placeholders = ", ".join("?" for _ in target_refs)
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT r.operation_id, r.operation_name, r.operation_version,
                       r.plan_hash, r.target_digest
                FROM operation_runs AS r
                JOIN operation_targets AS t ON t.operation_id = r.operation_id
                WHERE (r.status IN ('running', 'interrupted')
                       OR r.terminal_reason = 'recovery_unknown'
                       OR r.terminal_reason LIKE 'recovered_partial:%')
                  AND t.target_ref IN ({placeholders})
                ORDER BY r.started_at_ms, r.operation_id
                """,
                target_refs,
            ).fetchall()
            operations = [self._recovery_operation(conn, row) for row in rows]
        return tuple(operations)

    def attempt_owner_liveness(self, operation_id: str) -> Literal["dead", "live", "unknown"]:
        """Classify whether a nonterminal operation may be recovered.

        Unknown is intentionally distinct from dead.  A legacy owner string or
        unreadable process witness blocks recovery and overlap just like a live
        worker, because either can still own an in-flight target mutation.
        """

        with self._connection() as conn:
            owners = conn.execute(
                "SELECT worker_id FROM operation_attempts WHERE operation_id = ? AND state = 'running'",
                (operation_id,),
            ).fetchall()
        states = {_attempt_owner_liveness(cast(str | None, row[0])) for row in owners}
        if "live" in states:
            return "live"
        if "unknown" in states:
            return "unknown"
        return "dead"

    def orphaned_operations(self) -> tuple[RecoveryOperation, ...]:
        """Discover nonterminal operations whose owner is conclusively absent.

        A legacy/unreadable owner is intentionally not an orphan.  This is the
        liveness barrier that prevents recovery startup from stealing live
        work before a domain inspector sees it.
        """

        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT operation_id FROM operation_runs
                WHERE status IN ('running', 'interrupted') ORDER BY started_at_ms, operation_id
                """
            ).fetchall()
            operation_ids: list[str] = []
            for row in rows:
                operation_id = str(row[0])
                owners = conn.execute(
                    "SELECT worker_id FROM operation_attempts WHERE operation_id = ? AND state = 'running'",
                    (operation_id,),
                ).fetchall()
                if owners and not all(
                    _attempt_owner_liveness(cast(str | None, owner[0])) == "dead" for owner in owners
                ):
                    continue
                operation_ids.append(operation_id)
        recovered: list[RecoveryOperation] = []
        for operation_id in operation_ids:
            with self._connection() as conn:
                row = conn.execute(
                    """
                    SELECT operation_id, operation_name, operation_version, plan_hash, target_digest
                    FROM operation_runs WHERE operation_id = ? AND status IN ('running', 'interrupted')
                    """,
                    (operation_id,),
                ).fetchone()
                if row is None:
                    continue
                recovered.append(self._recovery_operation(conn, row))
        return tuple(recovered)

    @staticmethod
    def _recovery_operation(conn: sqlite3.Connection, row: sqlite3.Row | tuple[object, ...]) -> RecoveryOperation:
        """Reconstruct recovery targets without silently dropping damaged evidence.

        ``operation_targets`` is the durable effect identity. Preview rows add
        policy detail, but a missing preview target must make recovery unknown,
        never erase a target and turn a partial delete into a false success.
        """

        operation_id, operation, version, plan_hash, target_digest = row[:5]
        expected = int(
            conn.execute("SELECT target_count FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone()[
                0
            ]
        )
        target_rows = conn.execute(
            """
            SELECT t.target_kind, t.target_ref, t.identity_digest, t.effect_identity,
                   p.durability, p.recovery_policy
            FROM operation_targets AS t
            LEFT JOIN operation_preview_targets AS p
              ON p.preview_id = (SELECT preview_id FROM operation_runs WHERE operation_id = t.operation_id)
             AND p.ordinal = t.ordinal
            WHERE t.operation_id = ? ORDER BY t.ordinal
            """,
            (operation_id,),
        ).fetchall()
        context: Mapping[str, object] = {}
        preview = conn.execute(
            """
            SELECT p.plan_json
            FROM operation_previews AS p
            JOIN operation_runs AS r ON r.preview_id = p.preview_id
            WHERE r.operation_id = ?
            """,
            (operation_id,),
        ).fetchone()
        if preview is not None:
            try:
                plan_payload = json.loads(str(preview[0]))
            except (TypeError, json.JSONDecodeError):
                plan_payload = None
            if isinstance(plan_payload, dict) and isinstance(plan_payload.get("context"), dict):
                context = cast(Mapping[str, object], plan_payload["context"])
        complete = (
            expected > 0
            and len(target_rows) == expected
            and all(target[4] is not None and target[5] is not None for target in target_rows)
        )
        targets = (
            tuple(
                MutationTarget(
                    kind=str(target[0]),
                    ref=str(target[1]),
                    policy_key="audit-recovery",
                    identity_digest=str(target[2]),
                    effect_identity=str(target[3]),
                    durability=cast(Any, str(target[4])),
                    recovery=cast(Any, str(target[5])),
                )
                for target in target_rows
            )
            if complete
            else ()
        )
        return RecoveryOperation(
            operation_id=str(operation_id),
            operation=str(operation),
            operation_version=int(cast(Any, version)),
            plan_hash=str(plan_hash),
            target_digest=str(target_digest),
            targets=targets,
            expected_target_count=expected,
            reconstructed_target_count=len(target_rows),
            target_evidence_complete=complete,
            context=context,
        )

    @_continuity_mutation("record_recovery_disposition")
    def record_recovery_disposition(self, operation_id: str, disposition: RecoveryDisposition) -> None:
        """Finalize one orphaned operation from domain-provided target evidence."""

        now_ms = cast(int, self._command_value("now_ms", int(time.time() * 1000)))
        with self._connection() as conn:
            self._begin(conn)
            run = conn.execute(
                "SELECT actor_ref, status, terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if run is None:
                raise ValueError(f"unknown operation {operation_id!r}")
            status = str(run[1])
            # A run startup already terminalized as ``recovery_unknown`` is
            # still adjudicable evidence: a later adoption call (a rerun of a
            # committed recovery with matching intent/postflight/receipt) must
            # be able to finalize it, not just a freshly-dead ``interrupted``
            # run (polylogue-39pdi).
            if not _is_adjudicable_recovery(status, cast(str | None, run[2])):
                return
            live_owner = conn.execute(
                "SELECT worker_id FROM operation_attempts WHERE operation_id = ? AND state = 'running'", (operation_id,)
            ).fetchall()
            owner_liveness = {_attempt_owner_liveness(cast(str | None, row[0])) for row in live_owner}
            # A live or unverifiable owner may still be mutating its targets,
            # so it cannot be terminalized.  Refuse silently: the run stays
            # nonterminal, so it remains visible to overlap detection and to
            # ``recovery_status``/``adjudicate_recovery``.  Appending an event
            # here instead would grow the durable log on every restart and
            # every overlapping request without changing any state.
            if owner_liveness & {"live", "unknown"}:
                return
            target_rows = conn.execute(
                "SELECT target_ref FROM operation_targets WHERE operation_id = ? ORDER BY ordinal", (operation_id,)
            ).fetchall()
            expected_refs = {str(row[0]) for row in target_rows}
            per_target = {item.target_ref: item for item in disposition.target_dispositions}
            if disposition.kind != "unknown" and (
                not per_target
                or len(per_target) != len(disposition.target_dispositions)
                or set(per_target) != expected_refs
            ):
                raise ValueError("confirmed recovery disposition must classify every durable target exactly once")
            conn.execute(
                """
                UPDATE operation_attempts SET state = ?, finished_at_ms = ?, unknown_reason = ?
                WHERE operation_id = ? AND state IN ('running', 'unknown')
                """,
                (
                    "unknown" if disposition.kind == "unknown" else "reconciled",
                    now_ms,
                    disposition.detail,
                    operation_id,
                ),
            )
            if disposition.kind != "unknown":
                for target in per_target.values():
                    state = {"applied": "applied", "not-applied": "failed", "unknown": "unknown"}[target.state]
                    conn.execute(
                        """
                        UPDATE operation_targets
                        SET state = ?, completed_at_ms = ?, unknown_reason = ?, domain_receipt_ref = ?, domain_receipt_kind = ?
                        WHERE operation_id = ? AND target_ref = ? AND state IN ('pending', 'running', 'unknown')
                        """,
                        (
                            state,
                            now_ms if state != "unknown" else None,
                            target.detail or disposition.detail,
                            disposition.evidence_ref,
                            "recovery-evidence" if disposition.evidence_ref else None,
                            operation_id,
                            target.target_ref,
                        ),
                    )
            else:
                conn.execute(
                    """
                    UPDATE operation_targets SET state = ?, completed_at_ms = ?, unknown_reason = ?
                    WHERE operation_id = ? AND state IN ('pending', 'running', 'unknown')
                    """,
                    ("unknown", None, disposition.detail, operation_id),
                )
            states = [
                str(row[0])
                for row in conn.execute("SELECT state FROM operation_targets WHERE operation_id = ?", (operation_id,))
            ]
            if disposition.kind == "unknown" or "unknown" in states:
                run_state, reason = "failed", "recovery_unknown"
            elif disposition.kind == "confirmed-applied":
                run_state, reason = "completed", "recovered_applied"
            elif disposition.kind == "confirmed-not-applied":
                run_state, reason = "failed", "recovered_not_applied"
            else:
                run_state, reason = "failed", f"recovered_partial:{disposition.action}"
            conn.execute(
                """
                UPDATE operation_runs
                SET status = ?, terminal_reason = ?, updated_at_ms = ?, completed_at_ms = ?,
                    unknown_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'unknown'),
                    affected_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'applied'),
                    failed_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'failed'),
                    unknown_reason = ?
                WHERE operation_id = ?
                """,
                (
                    run_state,
                    reason,
                    now_ms,
                    now_ms,
                    operation_id,
                    operation_id,
                    operation_id,
                    disposition.detail,
                    operation_id,
                ),
            )
            self._append_event(
                conn,
                operation_id=operation_id,
                event_type="recovery_classified",
                from_state=str(run[1]),
                to_state=run_state,
                actor_ref=str(run[0]),
                occurred_at_ms=now_ms,
                detail={
                    "kind": disposition.kind,
                    "action": disposition.action,
                    "detail": (disposition.detail or "")[:512],
                    "evidence_ref": disposition.evidence_ref,
                    "targets": [
                        {"target_ref": item.target_ref, "state": item.state, "action": item.action}
                        for item in disposition.target_dispositions
                    ],
                },
            )

    def has_recovered_effect(self, plan: MutationPlan) -> str | None:
        """Return the durable recovery barrier for this exact semantic effect."""

        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT operation_id FROM operation_runs
                WHERE operation_name = ? AND operation_version = ?
                  AND archive_instance_id = ? AND archive_identity_digest = ?
                  AND parameter_digest = ? AND target_digest = ?
                  AND target_count > 0
                  AND terminal_reason = 'recovered_applied'
                ORDER BY completed_at_ms, operation_id LIMIT 1
                """,
                (
                    plan.operation,
                    plan.operation_version,
                    plan.archive_instance_id,
                    plan.archive_identity_digest,
                    plan.parameter_digest,
                    plan.target_digest,
                ),
            ).fetchone()
        return None if row is None else str(row[0])

    @_continuity_mutation("adjudicate_recovery")
    def adjudicate_recovery(
        self,
        operation_id: str,
        *,
        target_outcomes: Mapping[str, Literal["applied", "not-applied", "unknown"]],
        reason: str,
        adjudicator: str | None = None,
    ) -> None:
        """Apply an authorized, bounded per-target decision to an unknown run.

        ``adjudicator`` identifies who made this decision. It is recorded as
        the ``recovery_adjudicated`` event's actor -- the original mutation's
        ``actor_ref`` must never be reused there, since that would misattribute
        the adjudication decision to whoever ran the *original* mutation in
        the append-only audit trail (polylogue-39pdi). Leave unset only when
        no adjudicator identity is available.
        """

        if not reason.strip():
            raise ValueError("recovery adjudication requires a reason")
        now_ms = cast(int, self._command_value("now_ms", int(time.time() * 1000)))
        with self._connection() as conn:
            self._begin(conn)
            run = conn.execute(
                "SELECT actor_ref, status, terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if run is None:
                raise ValueError(f"operation {operation_id!r} is not awaiting recovery adjudication")
            status = str(run[1])
            adjudicable = _is_adjudicable_recovery(status, cast(str | None, run[2]))
            if not adjudicable:
                raise ValueError(f"operation {operation_id!r} is not awaiting recovery adjudication")
            if status == "running":
                # polylogue-39pdi: unlike the automatic recovery paths
                # (``orphaned_operations``/``recover_abandoned_attempts``),
                # adjudicating a ``running`` operation used to accept it
                # unconditionally. A live standalone executor (MCP or another
                # process with no daemon pidfile involved) could still be
                # applying the mutation while this call marks targets
                # reconciled and removes the overlap barrier -- permitting
                # concurrent/duplicate effects. Require the same conclusive
                # dead-owner evidence the automatic paths already demand.
                live_owner = conn.execute(
                    "SELECT worker_id FROM operation_attempts WHERE operation_id = ? AND state = 'running'",
                    (operation_id,),
                ).fetchall()
                owner_liveness = {_attempt_owner_liveness(cast(str | None, row[0])) for row in live_owner}
                if owner_liveness & {"live", "unknown"}:
                    raise ValueError(
                        f"operation {operation_id!r} attempt owner is not conclusively dead; "
                        "cannot adjudicate a running operation"
                    )
            targets = conn.execute(
                "SELECT ordinal, target_ref FROM operation_targets WHERE operation_id = ? ORDER BY ordinal",
                (operation_id,),
            ).fetchall()
            if len(targets) > 256:
                raise ValueError("recovery adjudication exceeds the 256-target command budget")
            expected = {str(target[1]) for target in targets}
            if set(target_outcomes) != expected:
                raise ValueError("recovery adjudication must name every durable target exactly once")
            for ordinal, target_ref in targets:
                outcome = target_outcomes[str(target_ref)]
                if outcome not in {"applied", "not-applied", "unknown"}:
                    raise ValueError("recovery adjudication outcomes must be applied, not-applied, or unknown")
                state = {"applied": "applied", "not-applied": "failed", "unknown": "unknown"}[outcome]
                conn.execute(
                    "UPDATE operation_targets SET state = ?, completed_at_ms = ?, unknown_reason = ? WHERE operation_id = ? AND ordinal = ?",
                    (state, now_ms if state != "unknown" else None, reason[:512], operation_id, int(ordinal)),
                )
            states = [
                str(row[0])
                for row in conn.execute("SELECT state FROM operation_targets WHERE operation_id = ?", (operation_id,))
            ]
            run_state, terminal_reason = _run_state_for_targets(states)
            if states and all(state == "applied" for state in states):
                # An operator's applied decision is still recovery evidence.
                # Keep the duplicate-effect barrier that an automatic
                # confirmed-applied classification would have installed.
                run_state, terminal_reason = "completed", "recovered_applied"
            if "unknown" in states:
                # An adjudicated unknown is still unknown: keep the run
                # terminal-but-adjudicable rather than reopening it as
                # interrupted, which a later startup would reclassify.
                run_state, terminal_reason = "failed", "recovery_unknown"
            conn.execute(
                """
                UPDATE operation_attempts SET state = ?, finished_at_ms = ?, unknown_reason = ?
                WHERE operation_id = ? AND state IN ('running', 'unknown')
                """,
                ("unknown" if "unknown" in states else "reconciled", now_ms, reason[:512], operation_id),
            )
            conn.execute(
                """
                UPDATE operation_runs SET status = ?, terminal_reason = ?, updated_at_ms = ?,
                    completed_at_ms = CASE WHEN ? IN ('completed', 'failed', 'interrupted') THEN ? ELSE completed_at_ms END,
                    unknown_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'unknown'),
                    affected_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'applied'),
                    failed_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'failed'),
                    unknown_reason = ? WHERE operation_id = ?
                """,
                (
                    run_state,
                    terminal_reason,
                    now_ms,
                    run_state,
                    now_ms,
                    operation_id,
                    operation_id,
                    operation_id,
                    reason[:512],
                    operation_id,
                ),
            )
            self._append_event(
                conn,
                operation_id=operation_id,
                event_type="recovery_adjudicated",
                from_state=str(run[1]),
                to_state=run_state,
                actor_ref=adjudicator,
                occurred_at_ms=now_ms,
                detail={"reason": reason[:512], "targets": dict(target_outcomes)},
            )

    @_continuity_mutation("recover_abandoned_attempts")
    def _recover_abandoned_attempts(self) -> tuple[str, ...]:
        """Mark only attempts whose recorded owner is no longer live as unknown."""

        now_ms = cast(int, self._command_value("now_ms", int(time.time() * 1000)))
        with self._connection() as conn:
            self._begin(conn)
            rows = conn.execute(
                "SELECT operation_id, worker_id FROM operation_attempts WHERE state = 'running' ORDER BY operation_id"
            ).fetchall()
            operation_ids = tuple(
                str(row[0]) for row in rows if _attempt_owner_liveness(cast(str | None, row[1])) == "dead"
            )
            for operation_id in operation_ids:
                conn.execute(
                    "UPDATE operation_attempts SET state = 'unknown', finished_at_ms = ?, unknown_reason = ? WHERE operation_id = ? AND state = 'running'",
                    (now_ms, "process ended before audit finalization", operation_id),
                )
                conn.execute(
                    "UPDATE operation_targets SET state = 'unknown', unknown_reason = ? WHERE operation_id = ? AND state IN ('running', 'pending')",
                    ("process ended before audit finalization", operation_id),
                )
                conn.execute(
                    "UPDATE operation_runs SET status = 'interrupted', terminal_reason = 'unknown_effect', updated_at_ms = ?, completed_at_ms = ?, unknown_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'unknown'), unknown_reason = ? WHERE operation_id = ?",
                    (now_ms, now_ms, operation_id, "process ended before audit finalization", operation_id),
                )
                self._append_event(
                    conn,
                    operation_id=operation_id,
                    event_type="attempt_unknown",
                    from_state="running",
                    to_state="interrupted",
                    occurred_at_ms=now_ms,
                    detail={"reason": "process ended before audit finalization"},
                )
        return operation_ids

    def get_operation(self, operation_id: str) -> dict[str, object] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone()
            return dict(row) if row is not None else None

    def find_interrupted_operation(self, *, operation_name: str, parameter_digest: str) -> str | None:
        """Return one dead audit attempt bound to an exact recovery plan.

        Matches both a freshly-dead ``interrupted`` run and one startup has
        already terminalized as ``recovery_unknown`` -- the latter is still
        adjudicable evidence, and a rerun of a committed raw recovery with
        matching intent/postflight/receipt evidence must be able to adopt it
        rather than staying wedged because startup got there first
        (polylogue-39pdi).
        """

        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT operation_id FROM operation_runs
                WHERE operation_name = ? AND parameter_digest = ?
                  AND (status = 'interrupted'
                       OR (status = 'failed' AND (terminal_reason = 'recovery_unknown'
                           OR terminal_reason LIKE 'recovered_partial:%')))
                ORDER BY started_at_ms, operation_id
                """,
                (operation_name, parameter_digest),
            ).fetchall()
        if len(rows) > 1:
            raise RuntimeError("multiple interrupted operations share the same durable parameter digest")
        return None if not rows else str(rows[0][0])

    def recovery_operation(self, operation_id: str) -> RecoveryOperation:
        """Load one recoverable operation with its durable target evidence."""

        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT operation_id, operation_name, operation_version, plan_hash, target_digest
                FROM operation_runs
                WHERE operation_id = ?
                  AND (status = 'interrupted' OR (status = 'failed' AND (terminal_reason = 'recovery_unknown'
                       OR terminal_reason LIKE 'recovered_partial:%')))
                """,
                (operation_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"operation {operation_id!r} is not recoverable")
            return self._recovery_operation(conn, row)

    def list_recovery_operations(self) -> tuple[dict[str, object], ...]:
        """List every interrupted or recovery-unknown run with target states.

        This is the discovery counterpart to :meth:`recovery_operation`.
        Recovery must remain bound to an exact operation id, but an operator
        must be able to obtain that id without already knowing the operation's
        name or parameter digest.
        """

        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM operation_runs
                WHERE status = 'interrupted'
                   OR (status = 'failed' AND (terminal_reason = 'recovery_unknown'
                       OR terminal_reason LIKE 'recovered_partial:%'))
                ORDER BY started_at_ms, operation_id
                """
            ).fetchall()
            return tuple(
                {"operation": dict(row), "targets": self._list_targets(conn, str(row["operation_id"]))} for row in rows
            )

    @staticmethod
    def _list_targets(conn: sqlite3.Connection, operation_id: str) -> tuple[dict[str, object], ...]:
        rows = conn.execute(
            """
            SELECT ordinal, target_ref, state, completed_at_ms, unknown_reason
            FROM operation_targets WHERE operation_id = ? ORDER BY ordinal
            """,
            (operation_id,),
        ).fetchall()
        return tuple(dict(row) for row in rows)

    def list_targets(self, operation_id: str) -> tuple[dict[str, object], ...]:
        """Return one operation's ordered target dispositions (ref + current state).

        Used by ``operation-recovery`` status/inspection output (CLI and MCP)
        so an operator can see exactly which targets exist and what state
        each is in -- the information needed to construct a bounded
        ``--target-outcome target_ref=applied|not-applied|unknown``
        adjudication call without going to ``audit.db`` by hand
        (polylogue-39pdi). Unlike ``recovery_operation``, this is not
        restricted to ``interrupted``/``recovery_unknown`` runs: it is read-only
        status, valid for any operation id that exists.
        """

        with self._connection() as conn:
            return self._list_targets(conn, operation_id)

    def list_events(self, operation_id: str) -> tuple[dict[str, object], ...]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM operation_events WHERE operation_id = ? ORDER BY sequence", (operation_id,)
            ).fetchall()
            return tuple(dict(row) for row in rows)

    @staticmethod
    def _append_event(
        conn: sqlite3.Connection,
        *,
        operation_id: str,
        event_type: str,
        occurred_at_ms: int,
        detail: Mapping[str, object],
        target_ordinal: int | None = None,
        attempt_id: str | None = None,
        from_state: str | None = None,
        to_state: str | None = None,
        actor_ref: str | None = None,
    ) -> None:
        sequence = int(
            conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM operation_events WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()[0]
        )
        conn.execute(
            """
            INSERT INTO operation_events(
                operation_id, sequence, target_ordinal, attempt_id, event_type,
                from_state, to_state, actor_ref, occurred_at_ms, detail_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                operation_id,
                sequence,
                target_ordinal,
                attempt_id,
                event_type,
                from_state,
                to_state,
                actor_ref,
                occurred_at_ms,
                json.dumps(dict(detail), sort_keys=True, separators=(",", ":")),
            ),
        )


__all__ = ["AuditRepository", "AuditTargetState", "token_sha256"]

"""Durable audit.db repository for mutation authority and lifecycle evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from polylogue.operations.ingest_acceptance import INGEST_OPERATION
from polylogue.operations.machine_receipts import (
    MAX_PAGE_ITEMS,
    IngestHistoricalReceiptV2,
    IngestInputPageHistoricalReceipt,
    IngestInputRawPageHistoricalReceipt,
    IngestInsightPageHistoricalReceipt,
    IngestRefusalPageHistoricalReceipt,
    IngestRefusedMembershipHistorical,
    MachineHistoricalReceipt,
    decode_machine_receipt,
    encode_machine_receipt,
    ingest_input_raw_pages_digest,
    ingest_insight_pages_digest,
    ingest_refusal_pages_digest,
    ingest_session_ids_digest,
)
from polylogue.operations.mutation_transaction import (
    DELETE_PREVIEW_SAMPLE_IDS,
    AuthorizationMismatchError,
    MutationAuthorization,
    MutationPlan,
    MutationPreview,
    MutationPrincipal,
    MutationReceipt,
    MutationTarget,
    RecoveryOperation,
    RecoveryResolution,
    TokenConsumedError,
    TokenExpiredError,
    validate_mutation_plan_integrity,
)
from polylogue.storage.sqlite.archive_tiers.source_items import (
    FrozenSourceManifest,
    SealedSourceManifestRef,
    source_manifest_from_dict,
)
from polylogue.storage.sqlite.audit_continuity import (
    EXCISION_SOURCE_COMMIT_KIND,
    AuditContinuityCoordinator,
    AuditContinuityUnknownMutationError,
    AuditMutation,
    CanonicalAuditLiteral,
    excision_completion_identity,
)
from polylogue.storage.sqlite.audit_continuity import AuditContinuityError as AuditContinuityError
from polylogue.storage.sqlite.audit_continuity import AuditContinuityPendingError as AuditContinuityPendingError
from polylogue.storage.sqlite.audit_leaf import (
    AuditLeafError,
    assert_verified_audit_leaf,
    open_verified_audit_connection,
    open_verified_audit_read_connection,
)

if TYPE_CHECKING:
    from polylogue.operations.daemon_protocol import DaemonOperationRequest
    from polylogue.operations.insight_acceptance import AcceptedInsightPart

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
_MAX_MACHINE_AUTHORITY_PARTS = 40

#: Machine batch transitions a request can accept in pages. One transition
#: carries at most :data:`_MAX_MACHINE_AUTHORITY_PARTS` parts (the bound on a
#: continuity payload); a request of any size appends pages under one staged
#: record, ``<kind>-pages``, and its final page turns it into ``<kind>``.
_PAGED_MACHINE_KINDS: dict[str, str] = {
    "create_preview_batch": "preview-batch",
    "issue_authorization_batch": "authorization-batch",
    "accept_execution_batch": "execution-batch",
    "cancel_preview_batch": "cancelled-preview-batch",
}


def machine_pages_kind(kind: str) -> str:
    """The staged artifact kind of a paged machine batch still accepting pages."""
    return f"{kind}-pages"


#: Parts one page of a paged machine batch carries.
MACHINE_PAGE_PARTS = _MAX_MACHINE_AUTHORITY_PARTS

#: How long a delete handshake may sit idle between one phase finishing and
#: the next being accepted before its authority clock resumes. While the
#: preview, authorization and execution phases follow one another within it,
#: expiry is judged as of the preview's acceptance, however long each phase
#: takes to page (``AuditRepository.handshake_as_of_ms``).
HANDSHAKE_IDLE_MS = 60_000

#: Staged kinds of paged machine batches.
MACHINE_PAGE_KINDS = frozenset(machine_pages_kind(kind) for kind in _PAGED_MACHINE_KINDS.values())
_MAX_INSIGHT_ACCEPTED_PARTS = 4096
_INSIGHT_MACHINE_OPERATION = "maintenance.insights.rebuild"


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


@dataclass(frozen=True, slots=True)
class AcceptedIdentityResetCustody:
    """Transient proof minted by relational Audit verification, never wire input."""

    repository: AuditRepository
    preview_ref: str
    plan_hash: str
    preview_request_id: str
    descendant: MachineRequestBinding
    ordinal: int
    principal: MutationPrincipal
    issued_at_ms: int
    source_part_count: int

    def require_authorization(self, preview: MutationPreview, principal: MutationPrincipal) -> None:
        if (
            preview.preview_ref != self.preview_ref
            or preview.plan.plan_hash != self.plan_hash
            or principal != self.principal
            or preview.plan.operation != "mutate-identity-reset"
        ):
            raise AuthorizationMismatchError("reset authority differs from its verified custody")
        self.repository._revalidate_identity_reset_authorization_proof(self, preview, principal)


class MachineRequestConflictError(ValueError):
    """An exchange identifier was already bound to different authenticated intent."""


class MachineRequestRecoveredError(RuntimeError):
    """Stop dispatch when a durable exchange already owns the domain effect."""

    def __init__(self, record: dict[str, object]) -> None:
        super().__init__("machine request already has a durable domain reference")
        self.record = record


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

    detail: dict[str, object] = {
        "status": status,
        "reason": (reason or "")[:512],
        "receipt_ref": None if receipt is None else receipt.receipt_ref,
        "target_count": 0 if receipt is None else len(receipt.target_refs),
        "affected_count": 0 if receipt is None else receipt.affected_count,
    }
    if receipt is not None and receipt.historical_receipt is not None:
        detail["historical_receipt"] = encode_machine_receipt(receipt.historical_receipt)
    return detail


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
            # Continuity is unavailable only while source.db or audit.db is
            # absent (e.g. before audit adoption); a present tier missing its
            # continuity half raises instead of reaching this branch.
            if not self._continuity.is_available():
                if self._machine_binding is not None:
                    raise RuntimeError("machine acceptance requires source-WAL audit continuity")
                return method(self, *args, **kwargs)
            payload = self._continuity_payload(kind, args, kwargs)
            if self._machine_binding is not None and self._machine_binding[1] == kind:
                binding = self._machine_binding[0]
                prior = self.machine_request(binding)
                continuing_insight_staging = (
                    kind == "append_insight_preview"
                    and binding.operation_name == _INSIGHT_MACHINE_OPERATION
                    and prior is not None
                    and prior.get("artifact_kind") == "insight-preview-pages"
                )
                continuing_insight_seal = (
                    kind == "seal_insight_execution"
                    and binding.operation_name == _INSIGHT_MACHINE_OPERATION
                    and prior is not None
                    and prior.get("artifact_kind") == "insight-preview-pages"
                )
                page = self._machine_page
                continuing_page = (
                    page is not None
                    and page[0] > 0
                    and prior is not None
                    and prior.get("artifact_kind") == machine_pages_kind(_PAGED_MACHINE_KINDS[kind])
                    and prior.get("part_count") == page[0]
                )
                if page is not None and page[0] > 0 and prior is None:
                    raise MachineRequestConflictError("a later machine page has no staged request")
                if (
                    prior is not None
                    and self._machine_part is None
                    and not (continuing_insight_staging or continuing_insight_seal or continuing_page)
                ):
                    raise MachineRequestRecoveredError(prior)
                payload["machine_request"] = binding.to_dict()
                if binding.operation_name in {
                    "mutation.identity-reset.authorize",
                    "mutation.identity-reset",
                } and kind in {"issue_authorization_batch", "accept_execution_batch"}:
                    if self._identity_reset_intent is None:
                        raise AuthorizationMismatchError("reset acceptance requires its admitted custody intent")
                    payload["identity_reset_intent"] = dict(self._identity_reset_intent)
                if page is not None:
                    payload["machine_page"] = {"offset": page[0], "final": page[1]}
                if self._machine_part is not None:
                    payload["machine_part"] = self._machine_part
                if self._machine_deadline_unix_ms is not None:
                    payload["accepted_deadline_unix_ms"] = self._machine_deadline_unix_ms
                # Insight page staging only preserves immutable preparation;
                # it is deliberately not an accepted execution.  The daemon's
                # peer/cancellation state may move to accepted only when the
                # sealed manifest is about to commit.  All existing machine
                # transitions retain their first-transition callback.
                insight_execution_seal = (
                    binding.operation_name == _INSIGHT_MACHINE_OPERATION and kind == "seal_insight_execution"
                )
                if self._before_machine_prepare is not None and (
                    insight_execution_seal or (prior is None and binding.operation_name != _INSIGHT_MACHINE_OPERATION)
                ):
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
    from polylogue.operations.machine_plan_context import replay_context

    payload = {
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
    semantic_context = replay_context(plan.operation, plan.context)
    if semantic_context is not None:
        payload["replay_context"] = semantic_context
    return payload


def plan_from_stored_payload(raw: object) -> MutationPlan:
    """Reconstruct a MutationPlan from its stored plan_json payload.

    The single owner of plan reconstruction. Every consumer that reads a
    persisted plan - preview loading, authorization resume, delete
    authorization - goes through here, so the rules cannot diverge per
    call site (polylogue-1ifhp).
    """
    return _plan_from_payload(raw)


def _plan_from_payload(raw: object) -> MutationPlan:
    value = cast(dict[str, object], raw)
    raw_context = value.get("context")
    if value.get("replay_context") is not None:
        from polylogue.operations.machine_plan_context import context_from_replay

        restored_context = context_from_replay(str(value["operation"]), value["replay_context"])
        if _context_sha256(restored_context) != value.get("context_sha256"):
            raise ValueError("machine replay context differs from its authored digest")
        if raw_context is not None and raw_context != restored_context:
            raise ValueError("stored plan context differs from its replay semantics")
        raw_context = restored_context
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
        "historical_receipt": (
            None if receipt.historical_receipt is None else encode_machine_receipt(receipt.historical_receipt)
        ),
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
        domain_receipt={},
        historical_receipt=(
            None if value.get("historical_receipt") is None else decode_machine_receipt(value["historical_receipt"])
        ),
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
        self._machine_page: tuple[int, bool] | None = None
        self._identity_reset_intent: Mapping[str, object] | None = None
        self._accepted_reset_custody: AcceptedIdentityResetCustody | None = None
        self._before_machine_prepare = before_machine_prepare
        self._on_commit = on_commit
        self._settled_reader = threading.local()

    @contextmanager
    def bind_machine_request(
        self,
        binding: MachineRequestBinding,
        *,
        transition: str,
        part: int | None = None,
        deadline_unix_ms: int | None = None,
        page: tuple[int, bool] | None = None,
    ) -> Iterator[None]:
        """Bind exactly one domain transition in its existing continuity transaction.

        ``page`` = (offset, final) appends one page of a paged batch
        transition at part ``offset``; the final page completes the request.
        """

        if transition not in {
            "create_preview",
            "issue_authorization",
            "consume_authorization_and_start",
            "cancel_preview",
            "create_preview_batch",
            "issue_authorization_batch",
            "cancel_preview_batch",
            "accept_execution_batch",
            "seal_insight_execution",
            "append_insight_preview",
            "accept_ingest",
        }:
            raise ValueError("machine request must bind a declared audit authority transition")
        if self._machine_binding is not None:
            raise RuntimeError("machine request binding scopes cannot overlap")
        # Parts are rows of the request; a paged batch has as many as it
        # accepted, so an ordinal is bounded only by what the request holds.
        if part is not None and (transition != "consume_authorization_and_start" or part < 0):
            raise ValueError("machine part must name an execution ordinal")
        if page is not None and (transition not in _PAGED_MACHINE_KINDS or page[0] < 0):
            raise ValueError("only a paged batch transition takes a page")
        self._machine_binding = (binding, transition)
        self._machine_part = part
        self._machine_deadline_unix_ms = deadline_unix_ms
        self._machine_page = page
        try:
            yield
        finally:
            self._machine_binding = None
            self._machine_part = None
            self._machine_deadline_unix_ms = None
            self._machine_page = None

    @contextmanager
    def bind_identity_reset_request(
        self,
        binding: MachineRequestBinding,
        admitted_request: DaemonOperationRequest,
        *,
        transition: str,
        page: tuple[int, bool],
    ) -> Iterator[None]:
        """Carry admitted reset intent on the existing outer continuity command."""
        expected = {
            "issue_authorization_batch": ("mutation.identity-reset.authorize", "preview_request_id"),
            "accept_execution_batch": ("mutation.identity-reset", "authorization_request_id"),
        }.get(transition)
        if (
            expected is None
            or admitted_request.operation != expected[0]
            or admitted_request.request_id != binding.request_id
            or admitted_request.fingerprint != binding.fingerprint
            or binding.operation_name != admitted_request.operation
        ):
            raise AuthorizationMismatchError("reset custody differs from the admitted request")
        source = admitted_request.payload.get(expected[1])
        if not isinstance(source, str) or (
            expected[0].endswith(".authorize") and admitted_request.payload.get("confirm") is not True
        ):
            raise AuthorizationMismatchError("reset custody requires its confirmed source request")
        descriptor = {
            "format": "identity-reset-custody/v1",
            "source_request_id": source,
            "request_intent": admitted_request.fingerprint_intent,
        }
        with self.bind_machine_request(binding, transition=transition, page=page):
            self._identity_reset_intent = descriptor
            try:
                yield
            finally:
                self._identity_reset_intent = None

    def require_accepted_identity_reset_custody(
        self,
        preview: MutationPreview,
        principal: MutationPrincipal,
        *,
        ordinal: int,
        issued_at_ms: int,
    ) -> AcceptedIdentityResetCustody:
        """Mint a transient proof from live request custody and durable preview rows."""
        if self._machine_binding is None or self._identity_reset_intent is None or self._machine_page is None:
            raise AuthorizationMismatchError("reset authorization requires accepted request custody")
        binding, transition = self._machine_binding
        command = {
            "machine_request": binding.to_dict(),
            "machine_page": {"offset": self._machine_page[0], "final": self._machine_page[1]},
            "identity_reset_intent": self._identity_reset_intent,
        }
        with self._connection() as conn:
            return self._verify_identity_reset_custody(
                conn,
                command=command,
                transition=transition,
                ordinal=ordinal,
                preview=preview,
                principal=principal,
                issued_at_ms=issued_at_ms,
            )

    def _verify_identity_reset_custody(
        self,
        conn: sqlite3.Connection,
        *,
        command: Mapping[str, object],
        transition: str,
        ordinal: int,
        preview: MutationPreview,
        principal: MutationPrincipal,
        issued_at_ms: int,
        authorization_ref: str | None = None,
    ) -> AcceptedIdentityResetCustody:
        """Re-prove each reset part at the normal/replay persistence boundary."""
        from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL, DaemonOperationRequest

        raw_binding = command.get("machine_request")
        if not isinstance(raw_binding, dict):
            raise AuthorizationMismatchError("reset custody lacks its machine request")
        binding = MachineRequestBinding(
            *(
                str(raw_binding[key])
                for key in ("archive_identity", "request_id", "principal_ref", "fingerprint", "operation_name")
            )
        )
        if binding.principal_ref != principal.actor_ref or preview.plan.operation != "mutate-identity-reset":
            raise AuthorizationMismatchError("reset custody differs from its principal or operation")
        current_row = conn.execute(
            "SELECT * FROM machine_requests WHERE request_id=?", (binding.request_id,)
        ).fetchone()
        current = None if current_row is None else dict(current_row)
        if current is not None and any(
            current[key] != value for key, value in binding.to_dict().items() if key != "archive_identity"
        ):
            raise AuthorizationMismatchError("reset custody request belongs to another principal or intent")
        source: dict[str, object] | None
        source_kind: str
        source_operation: str
        if transition == "consume_authorization_and_start":
            if (
                binding.operation_name != "mutation.identity-reset"
                or current is None
                or current["artifact_kind"] != "execution-batch"
                or current.get("stop_reason") is not None
                or command.get("machine_part") != ordinal
                or authorization_ref is None
            ):
                raise AuthorizationMismatchError("reset execution is not a sealed unconsumed request part")
            reserved = conn.execute(
                "SELECT authorization_ref,preview_ref,operation_id FROM machine_request_parts "
                "WHERE archive_identity=? AND request_id=? AND ordinal=?",
                (binding.archive_identity, binding.request_id, ordinal),
            ).fetchone()
            if reserved is None or tuple(reserved) != (authorization_ref, preview.preview_ref, None):
                raise AuthorizationMismatchError("reset execution reservation differs from its exact part")
            sources = conn.execute(
                "SELECT r.* FROM machine_request_parts p JOIN machine_requests r "
                "ON r.archive_identity=p.archive_identity AND r.request_id=p.request_id "
                "WHERE p.artifact_ref=? AND r.operation_name='mutation.identity-reset.authorize' "
                "AND r.artifact_kind='authorization-batch'",
                (authorization_ref,),
            ).fetchall()
            if len(sources) != 1:
                raise AuthorizationMismatchError("reset execution lacks sealed confirmation custody")
            source = dict(sources[0])
            source_kind, source_operation = "authorization-batch", "mutation.identity-reset.authorize"
        else:
            expected = {
                "issue_authorization_batch": (
                    "mutation.identity-reset.authorize",
                    "preview_request_id",
                    "preview-batch",
                    "mutation.identity-reset.preview",
                    "authorization-batch-pages",
                ),
                "accept_execution_batch": (
                    "mutation.identity-reset",
                    "authorization_request_id",
                    "authorization-batch",
                    "mutation.identity-reset.authorize",
                    "execution-batch-pages",
                ),
            }.get(transition)
            descriptor = command.get("identity_reset_intent")
            page = command.get("machine_page")
            if (
                expected is None
                or not isinstance(descriptor, dict)
                or not isinstance(page, dict)
                or descriptor.get("format") != "identity-reset-custody/v1"
                or set(descriptor) != {"format", "source_request_id", "request_intent"}
                or not isinstance(descriptor.get("request_intent"), dict)
            ):
                raise AuthorizationMismatchError("reset custody intent is unavailable")
            request = DaemonOperationRequest.from_dict(
                {
                    **descriptor["request_intent"],
                    "request_id": binding.request_id,
                    "protocol": DAEMON_OPERATION_PROTOCOL,
                }
            )
            source_id = descriptor.get("source_request_id")
            offset = page.get("offset")
            if (
                request.operation != expected[0]
                or request.operation != binding.operation_name
                or request.fingerprint != binding.fingerprint
                or request.payload.get(expected[1]) != source_id
                or (transition == "issue_authorization_batch" and request.payload.get("confirm") is not True)
                or type(offset) is not int
                or type(page.get("final")) is not bool
                or ordinal < offset
                or ordinal >= offset + MACHINE_PAGE_PARTS
            ):
                raise AuthorizationMismatchError("reset custody intent differs from its exact admitted part")
            if current is None:
                if offset != 0:
                    raise AuthorizationMismatchError("reset custody has no earlier staged page")
            elif (
                current["artifact_kind"] != expected[4]
                or current["part_count"] != offset
                or current.get("stop_reason") is not None
            ):
                raise AuthorizationMismatchError("reset custody staged request is stopped or inconsistent")
            source_row = conn.execute(
                "SELECT * FROM machine_requests WHERE request_id=? AND principal_ref=?",
                (str(source_id), principal.actor_ref),
            ).fetchone()
            source = None if source_row is None else dict(source_row)
            source_kind, source_operation = expected[2], expected[3]
        if (
            source is None
            or source["artifact_kind"] != source_kind
            or source["operation_name"] != source_operation
            or source["principal_ref"] != principal.actor_ref
            or source.get("stop_reason") is not None
            or ordinal < 0
            or ordinal >= int(cast(int, source["part_count"]))
        ):
            raise AuthorizationMismatchError("reset custody source is not a sealed principal-owned phase")
        source_part = conn.execute(
            "SELECT p.preview_ref,COALESCE(p.authorization_ref,p.artifact_ref),v.plan_hash,v.operation_name,v.principal_actor_ref,"
            "v.principal_surface,v.role_label,v.state FROM machine_request_parts p "
            "JOIN operation_previews v ON v.preview_id=p.preview_ref "
            "WHERE p.archive_identity=? AND p.request_id=? AND p.ordinal=?",
            (source["archive_identity"], source["request_id"], ordinal),
        ).fetchone()
        if (
            source_part is None
            or source_part[0] != preview.preview_ref
            or source_part[2] != preview.plan.plan_hash
            or tuple(source_part[3:7])
            != ("mutate-identity-reset", principal.actor_ref, principal.surface, principal.role_label)
            or source_part[7] != "prepared"
            or (authorization_ref is not None and source_part[1] != authorization_ref)
        ):
            raise AuthorizationMismatchError("reset custody does not match its frozen preview and principal")
        origins = conn.execute(
            "SELECT r.* FROM machine_request_parts p JOIN machine_requests r "
            "ON r.archive_identity=p.archive_identity AND r.request_id=p.request_id "
            "WHERE p.preview_ref=? AND p.ordinal=? AND r.operation_name='mutation.identity-reset.preview' "
            "AND r.artifact_kind='preview-batch'",
            (preview.preview_ref, ordinal),
        ).fetchall()
        if (
            len(origins) != 1
            or origins[0]["principal_ref"] != principal.actor_ref
            or origins[0]["stop_reason"] is not None
        ):
            raise AuthorizationMismatchError("reset custody originating preview is unavailable or cancelled")
        if issued_at_ms < 0:
            raise AuthorizationMismatchError("reset custody issuance time is invalid")
        return AcceptedIdentityResetCustody(
            self,
            preview.preview_ref,
            preview.plan.plan_hash,
            str(origins[0]["request_id"]),
            binding,
            ordinal,
            principal,
            issued_at_ms,
            int(origins[0]["part_count"]),
        )

    def _revalidate_identity_reset_authorization_proof(
        self,
        proof: AcceptedIdentityResetCustody,
        preview: MutationPreview,
        principal: MutationPrincipal,
    ) -> None:
        fresh = self.require_accepted_identity_reset_custody(
            preview,
            principal,
            ordinal=proof.ordinal,
            issued_at_ms=proof.issued_at_ms,
        )
        if fresh != proof:
            raise AuthorizationMismatchError("reset authorization custody changed")

    def handshake_as_of_ms(
        self,
        binding: MachineRequestBinding,
        *,
        preview_refs: tuple[str, ...] = (),
        authorization_refs: tuple[str, ...] = (),
    ) -> int:
        """The time a phase of a paged delete handshake is judged as of.

        A phase is judged as of its own durable acceptance, so a phase that
        began in time finishes however many pages it takes. When it consumes
        the artifacts of a completed earlier phase and was accepted within
        :data:`HANDSHAKE_IDLE_MS` of that phase's last page, it inherits the
        earlier phase's time instead: authority expires only while the
        handshake sits idle, not while it progresses.
        """
        record = self.machine_request(binding)
        accepted = int(cast(int, record["accepted_at_ms"])) if record is not None else int(time.time() * 1000)
        with self._connection() as conn:
            return self._phase_as_of(conn, accepted, preview_refs, authorization_refs)

    def _phase_as_of(
        self,
        conn: sqlite3.Connection,
        accepted_at_ms: int,
        preview_refs: tuple[str, ...],
        authorization_refs: tuple[str, ...],
    ) -> int:
        refs, kind, completion_sql = (
            (
                authorization_refs,
                "authorization-batch",
                """SELECT MAX(a.issued_at_ms) FROM machine_request_parts AS p
                   JOIN operation_authorizations AS a ON a.authorization_id = p.artifact_ref
                   WHERE p.archive_identity = ? AND p.request_id = ?""",
            )
            if authorization_refs
            else (
                preview_refs,
                "preview-batch",
                """SELECT MAX(v.created_at_ms) FROM machine_request_parts AS p
                   JOIN operation_previews AS v ON v.preview_id = p.artifact_ref
                   WHERE p.archive_identity = ? AND p.request_id = ?""",
            )
        )
        if not refs:
            return accepted_at_ms
        placeholders = ",".join("?" for _ in refs)
        rows = conn.execute(
            f"""SELECT DISTINCT r.archive_identity, r.request_id, r.accepted_at_ms
                FROM machine_request_parts AS p JOIN machine_requests AS r
                  ON r.archive_identity = p.archive_identity AND r.request_id = p.request_id
                WHERE p.artifact_ref IN ({placeholders}) AND r.artifact_kind = ?""",
            (*refs, kind),
        ).fetchall()
        if len(rows) != 1:
            # Not the artifacts of exactly one completed phase: wall time.
            return accepted_at_ms
        earlier_identity, earlier_request, earlier_accepted = rows[0]
        completed = conn.execute(completion_sql, (earlier_identity, earlier_request)).fetchone()[0]
        if completed is None or accepted_at_ms - int(completed) > HANDSHAKE_IDLE_MS:
            return accepted_at_ms
        earlier_previews: tuple[str, ...] = ()
        if authorization_refs:
            earlier_previews = tuple(
                str(row[0])
                for row in conn.execute(
                    """SELECT preview_ref FROM machine_request_parts
                       WHERE archive_identity = ? AND request_id = ? ORDER BY ordinal LIMIT ?""",
                    (earlier_identity, earlier_request, _MAX_MACHINE_AUTHORITY_PARTS),
                )
            )
        return min(accepted_at_ms, self._phase_as_of(conn, int(earlier_accepted), earlier_previews, ()))

    def machine_request(self, binding: MachineRequestBinding) -> dict[str, object] | None:
        """Recover the immutable domain reference and reject conflicting reuse.

        polylogue-cois9: ``machine_requests.archive_identity`` carries a live
        archive identity digest that folds in the *rebuildable* index tier's
        inode, so an ordinary index-generation promotion moves it.  Keying the
        recovery lookup on that column made a promotion silently lose request
        dedup: the row was simply unfindable and the retried request re-executed
        as new.  This audit database *is* the durable archive scope -- it lives
        inside the archive file set, and ``request_id`` is unique within it --
        so the durable key is ``request_id`` alone.  The stored identity is
        returned in the record so the caller can rebind its durable key to the
        one the surviving rows (and their ``machine_request_parts``) use.
        """

        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM machine_requests WHERE request_id = ?",
                (binding.request_id,),
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        expected = binding.to_dict()
        # ``archive_identity`` is deliberately excluded: a moved index
        # generation is not conflicting intent.  Principal, fingerprint and
        # operation still have to match exactly.
        del expected["archive_identity"]
        if any(record[key] != value for key, value in expected.items()):
            raise MachineRequestConflictError("request id is bound to another principal or intent")
        return record

    def machine_request_for_principal(
        self, archive_identity: str, request_id: str, principal_ref: str
    ) -> dict[str, object] | None:
        """Resolve a control reference only in the authenticated current archive.

        The archive scope is this audit database, not the caller's live archive
        identity digest -- see :meth:`machine_request`.  ``archive_identity``
        remains in the signature because callers authenticate it upstream, but
        it must not narrow the lookup, or a control reference disappears the
        moment a new index generation is promoted.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM machine_requests WHERE request_id = ? AND principal_ref = ?",
                (request_id, principal_ref),
            ).fetchone()
        return None if row is None else dict(row)

    @contextmanager
    def recovery_discovery_read(self) -> Iterator[None]:
        """Read recovery facts in their actual coordinated or settled view."""
        if self._coordinated_connection is not None:
            from polylogue.core.write_lease import require_write_lease

            if require_write_lease("coordinated recovery discovery", archive_root=self.path.parent) is None:
                raise RuntimeError("coordinated recovery discovery requires its original archive writer")
            yield
            return
        with self.settled_machine_read(wait_for_lock=True):
            yield

    @contextmanager
    def settled_machine_read(self, *, wait_for_lock: bool = False) -> Iterator[dict[str, int]]:
        """Read machine receipts without contending for audit's writer leaf.

        Continuity first proves that source and audit agree.  Every repository
        query nested in this scope then opens the verified WAL-aware read path;
        a completion waiter must never turn a concurrent durable mutation into
        a second writer acquisition.
        """
        with self._continuity.settled_read(wait_for_lock=wait_for_lock) as versions:
            depth = getattr(self._settled_reader, "depth", 0)
            self._settled_reader.depth = depth + 1
            try:
                yield versions
            finally:
                self._settled_reader.depth = depth

    def preview_for_principal(self, preview_ref: str, principal: MutationPrincipal) -> MutationPreview:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT plan_json, principal_actor_ref, principal_surface FROM operation_previews WHERE preview_id = ?",
                (preview_ref,),
            ).fetchone()
        if row is None or row[1] != principal.actor_ref or row[2] != principal.surface:
            raise AuthorizationMismatchError("preview does not belong to the authenticated principal")
        return MutationPreview(preview_ref=preview_ref, plan=_plan_from_payload(json.loads(row[0])))

    def identity_reset_preview_target_page(
        self,
        preview_request_id: str,
        principal: MutationPrincipal,
        *,
        archive_identity: str,
        offset: int,
        page_size: int,
    ) -> tuple[tuple[str, ...], int]:
        """Seek bounded target relations belonging to a complete reset selection.

        The publisher fills each plan to MUTATION_PLAN_PAGE_SIZE, except its
        final part. Offset therefore selects a part and its local target ordinal,
        never an OFFSET scan over the full target population.
        """
        from polylogue.operations.mutation_transaction import MUTATION_PLAN_PAGE_SIZE

        if offset < 0 or page_size < 1:
            raise ValueError("invalid identity reset target page")
        record = self.machine_request_for_principal(archive_identity, preview_request_id, principal.actor_ref)
        if (
            record is None
            or record["operation_name"] != "mutation.identity-reset.preview"
            or record["artifact_kind"] != "preview-batch"
            or record.get("stop_reason") is not None
        ):
            raise AuthorizationMismatchError("reset selection is not a sealed principal-owned preview batch")
        coordinates = (str(record["archive_identity"]), str(record["request_id"]))
        with self._connection() as conn:
            last = conn.execute(
                "SELECT p.ordinal,v.target_count,v.operation_name,v.principal_actor_ref,v.principal_surface "
                "FROM machine_request_parts p JOIN operation_previews v ON v.preview_id=p.preview_ref "
                "WHERE p.archive_identity=? AND p.request_id=? ORDER BY p.ordinal DESC LIMIT 1",
                coordinates,
            ).fetchone()
            if (
                last is None
                or int(last[0]) + 1 != record["part_count"]
                or not 0 <= int(last[1]) <= MUTATION_PLAN_PAGE_SIZE
                or tuple(last[2:]) != ("mutate-identity-reset", principal.actor_ref, principal.surface)
            ):
                raise AuthorizationMismatchError("reset selection's final immutable part is unavailable")
            total = int(last[0]) * MUTATION_PLAN_PAGE_SIZE + int(last[1])
            first = conn.execute(
                "SELECT v.principal_actor_ref,v.principal_surface FROM machine_request_parts p "
                "JOIN operation_previews v ON v.preview_id=p.preview_ref "
                "WHERE p.archive_identity=? AND p.request_id=? AND p.ordinal=0",
                coordinates,
            ).fetchone()
            if first is None or tuple(first) != (principal.actor_ref, principal.surface):
                raise AuthorizationMismatchError("reset selection belongs to another principal")
            part_ordinal, target_ordinal = divmod(min(offset, total), MUTATION_PLAN_PAGE_SIZE)
            remaining = min(page_size, max(0, total - offset))
            ids: list[str] = []
            while remaining:
                part = conn.execute(
                    "SELECT v.preview_id,v.operation_name,v.principal_actor_ref,v.principal_surface,v.target_count "
                    "FROM machine_request_parts p JOIN operation_previews v ON v.preview_id=p.preview_ref "
                    "WHERE p.archive_identity=? AND p.request_id=? AND p.ordinal=?",
                    (*coordinates, part_ordinal),
                ).fetchone()
                if (
                    part is None
                    or tuple(part[1:4]) != ("mutate-identity-reset", principal.actor_ref, principal.surface)
                    or int(part[4]) > MUTATION_PLAN_PAGE_SIZE
                ):
                    raise AuthorizationMismatchError("reset selection part is invalid")
                rows = conn.execute(
                    "SELECT target_ref FROM operation_preview_targets WHERE preview_id=? AND ordinal>=? "
                    "ORDER BY ordinal LIMIT ?",
                    (str(part[0]), target_ordinal, remaining),
                ).fetchall()
                if not rows:
                    raise ValueError("reset selection target population is incomplete")
                ids.extend(str(row[0]).removeprefix("session:") for row in rows)
                remaining -= len(rows)
                part_ordinal += 1
                target_ordinal = 0
        return tuple(ids), total

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

    def active_authorization_for_preview(
        self, preview_ref: str, principal: MutationPrincipal
    ) -> MutationAuthorization | None:
        """Return one reusable staged authority, refusing ambiguous duplicates."""

        with self._connection() as conn:
            rows = conn.execute(
                """SELECT authorization_id FROM operation_authorizations
                WHERE preview_id = ? AND actor_ref = ? AND surface = ? AND state = 'active'
                ORDER BY authorization_id""",
                (preview_ref, principal.actor_ref, principal.surface),
            ).fetchall()
        if len(rows) > 1:
            raise AuthorizationMismatchError("staged insight preview has ambiguous active authorizations")
        if not rows:
            return None
        _, authorization = self.authorization_for_principal(str(rows[0][0]), principal)
        return authorization

    def has_authorization_for_preview(self, preview_ref: str) -> bool:
        """Distinguish an unissued staged page from expired/consumed authority."""

        with self._connection() as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM operation_authorizations WHERE preview_id = ? LIMIT 1", (preview_ref,)
                ).fetchone()
                is not None
            )

    @staticmethod
    def _bind_machine_result(conn: sqlite3.Connection, mutation: AuditMutation, result: object) -> None:
        if mutation.kind == EXCISION_SOURCE_COMMIT_KIND:
            return
        raw = mutation.mapping_payload.get("machine_request")
        if raw is None:
            return
        binding = MachineRequestBinding(**cast(dict[str, str], raw))
        if mutation.kind == "accept_ingest":
            manifest = source_manifest_from_dict(mutation.mapping_payload["manifest"])
            conn.execute(
                """INSERT INTO machine_requests(
                    archive_identity, request_id, principal_ref, fingerprint, operation_name,
                    artifact_kind, artifact_ref, accepted_at_ms, accepted_deadline_unix_ms
                ) VALUES (?, ?, ?, ?, ?, 'source-generation', ?, ?, ?)""",
                (
                    *binding.to_dict().values(),
                    manifest.source_generation_id,
                    mutation.created_at_ms,
                    mutation.mapping_payload.get("accepted_deadline_unix_ms"),
                ),
            )
            operation_id = mutation.mapping_payload.get("ingest_operation_id")
            if operation_id is not None:
                preview_id = cast(str, mutation.mapping_payload["preview_id"])
                authorization_id = cast(str, mutation.mapping_payload["authorization_id"])
                conn.execute(
                    """INSERT INTO machine_request_parts(
                        archive_identity, request_id, ordinal, artifact_ref, preview_ref, authorization_ref, operation_id
                    ) VALUES (?, ?, 0, ?, ?, ?, ?)""",
                    (
                        binding.archive_identity,
                        binding.request_id,
                        manifest.source_generation_id,
                        preview_id,
                        authorization_id,
                        operation_id,
                    ),
                )
            return
        if mutation.kind == "append_insight_preview":
            if not isinstance(result, str):
                raise RuntimeError("insight preview staging requires a durable preview reference")
            prior = conn.execute(
                "SELECT artifact_kind, part_count FROM machine_requests WHERE archive_identity = ? AND request_id = ?",
                (binding.archive_identity, binding.request_id),
            ).fetchone()
            if prior is None:
                ordinal = 0
                conn.execute(
                    """INSERT INTO machine_requests(
                        archive_identity, request_id, principal_ref, fingerprint, operation_name,
                        artifact_kind, artifact_ref, accepted_at_ms, part_count, accepted_deadline_unix_ms
                    ) VALUES (?, ?, ?, ?, ?, 'insight-preview-pages', ?, ?, 1, ?)""",
                    (
                        *binding.to_dict().values(),
                        result,
                        mutation.created_at_ms,
                        mutation.mapping_payload.get("accepted_deadline_unix_ms"),
                    ),
                )
            else:
                if prior[0] != "insight-preview-pages" or int(prior[1]) >= _MAX_INSIGHT_ACCEPTED_PARTS:
                    raise MachineRequestConflictError("insight preview staging request cannot accept another page")
                ordinal = int(prior[1])
                conn.execute(
                    """UPDATE machine_requests SET part_count = part_count + 1
                    WHERE archive_identity = ? AND request_id = ? AND artifact_kind = 'insight-preview-pages'""",
                    (binding.archive_identity, binding.request_id),
                )
            conn.execute(
                """INSERT INTO machine_request_parts(
                    archive_identity, request_id, ordinal, artifact_ref, preview_ref, authorization_ref
                ) VALUES (?, ?, ?, ?, ?, NULL)""",
                (binding.archive_identity, binding.request_id, ordinal, result, result),
            )
            return
        if "machine_part" in mutation.mapping_payload:
            changed = conn.execute(
                """UPDATE machine_request_parts SET operation_id = ?
                WHERE archive_identity = ? AND request_id = ? AND ordinal = ? AND operation_id IS NULL""",
                (result, binding.archive_identity, binding.request_id, mutation.mapping_payload["machine_part"]),
            ).rowcount
            if changed != 1:
                raise MachineRequestConflictError("machine execution part has already started or is missing")
            return
        if mutation.kind in {
            "create_preview_batch",
            "issue_authorization_batch",
            "cancel_preview_batch",
            "accept_execution_batch",
            "seal_insight_execution",
        }:
            AuditRepository._bind_machine_batch(conn, mutation, binding, result)
            return
        plan = cast(
            dict[str, object],
            mutation.mapping_payload.get("plan")
            or cast(dict[str, object], mutation.mapping_payload["preview"])["plan"],
        )
        principal = mutation.mapping_payload.get("principal") or mutation.mapping_payload.get("authorization")
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
            artifact = cast(dict[str, object], mutation.mapping_payload["preview"])["preview_ref"]
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
        insight = binding.operation_name == _INSIGHT_MACHINE_OPERATION and mutation.kind == "seal_insight_execution"
        limit = _MAX_INSIGHT_ACCEPTED_PARTS if insight else _MAX_MACHINE_AUTHORITY_PARTS
        if not refs or len(refs) > limit:
            raise ValueError(f"machine authority batch must contain between 1 and {limit} parts")
        kind = {
            "create_preview_batch": "preview-batch",
            "issue_authorization_batch": "authorization-batch",
            "cancel_preview_batch": "cancelled-preview-batch",
            "accept_execution_batch": "execution-batch",
            "seal_insight_execution": "execution-batch",
        }[mutation.kind]
        raw_page = mutation.mapping_payload.get("machine_page")
        offset = 0
        if isinstance(raw_page, dict):
            offset, final = int(raw_page["offset"]), bool(raw_page["final"])
            staged = machine_pages_kind(kind)
            if offset == 0:
                conn.execute(
                    """INSERT INTO machine_requests(
                        archive_identity, request_id, principal_ref, fingerprint, operation_name,
                        artifact_kind, artifact_ref, accepted_at_ms, part_count, accepted_deadline_unix_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        *binding.to_dict().values(),
                        kind if final else staged,
                        refs[0],
                        mutation.created_at_ms,
                        len(refs),
                        mutation.mapping_payload.get("accepted_deadline_unix_ms"),
                    ),
                )
            else:
                changed = conn.execute(
                    """UPDATE machine_requests SET part_count = part_count + ?, artifact_kind = ?
                    WHERE archive_identity = ? AND request_id = ? AND artifact_kind = ? AND part_count = ?""",
                    (
                        len(refs),
                        kind if final else staged,
                        binding.archive_identity,
                        binding.request_id,
                        staged,
                        offset,
                    ),
                ).rowcount
                if changed != 1:
                    raise MachineRequestConflictError("machine page does not continue its staged request")
        elif insight:
            changed = conn.execute(
                """UPDATE machine_requests SET artifact_kind = ?, artifact_ref = ?, accepted_at_ms = ?,
                    accepted_deadline_unix_ms = ?
                WHERE archive_identity = ? AND request_id = ?
                  AND artifact_kind = 'insight-preview-pages' AND part_count = ?""",
                (
                    kind,
                    refs[0],
                    mutation.created_at_ms,
                    mutation.mapping_payload.get("accepted_deadline_unix_ms"),
                    binding.archive_identity,
                    binding.request_id,
                    len(refs),
                ),
            ).rowcount
            if changed != 1:
                raise MachineRequestConflictError("insight seal does not match its staged immutable page chain")
        else:
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
                    mutation.mapping_payload.get("accepted_deadline_unix_ms"),
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
            if insight:
                changed = conn.execute(
                    """UPDATE machine_request_parts SET artifact_ref = ?, authorization_ref = ?
                    WHERE archive_identity = ? AND request_id = ? AND ordinal = ?
                      AND preview_ref = ? AND authorization_ref IS NULL AND operation_id IS NULL""",
                    (ref, authorization_ref, binding.archive_identity, binding.request_id, ordinal, preview_ref),
                ).rowcount
                if changed != 1:
                    raise MachineRequestConflictError("insight seal page differs from its staged preview reference")
            else:
                conn.execute(
                    """INSERT INTO machine_request_parts(
                        archive_identity, request_id, ordinal, artifact_ref, preview_ref, authorization_ref
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        binding.archive_identity,
                        binding.request_id,
                        offset + ordinal,
                        ref,
                        preview_ref,
                        authorization_ref,
                    ),
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

    def sealed_insight_parts(
        self, binding: MachineRequestBinding, principal: MutationPrincipal
    ) -> tuple[AcceptedInsightPart, ...]:
        """Reload the exact immutable insights manifest already sealed to a request."""

        from polylogue.operations.insight_acceptance import (
            accepted_part_from_plan,
            insight_manifest_digest,
        )

        if binding.operation_name != _INSIGHT_MACHINE_OPERATION:
            raise ValueError("machine request is not an insights maintenance request")
        record = self.machine_request(binding)
        if record is None or record.get("artifact_kind") != "execution-batch":
            raise ValueError("insight manifest is not sealed")
        raw_parts = self.machine_parts(binding)
        part_count = record.get("part_count")
        if type(part_count) is not int or part_count != len(raw_parts) or not raw_parts:
            raise AuthorizationMismatchError("sealed insight manifest has incomplete machine parts")
        parts: list[AcceptedInsightPart] = []
        for ordinal, raw in enumerate(raw_parts):
            raw_ordinal = raw["ordinal"]
            if type(raw_ordinal) is not int or raw_ordinal != ordinal or raw["authorization_ref"] is None:
                raise AuthorizationMismatchError("sealed insight manifest part is incomplete")
            preview = self.preview_for_principal(str(raw["preview_ref"]), principal)
            if (
                preview.plan.operation != "mutate-rebuild-insights"
                or preview.plan.archive_identity_digest != binding.archive_identity
            ):
                raise AuthorizationMismatchError("sealed insight preview differs from request operation or archive")
            authorization_preview, _ = self.authorization_for_principal(str(raw["authorization_ref"]), principal)
            if authorization_preview.preview_ref != preview.preview_ref:
                raise AuthorizationMismatchError("sealed insight authorization differs from its preview page")
            parts.append(
                accepted_part_from_plan(
                    preview.plan,
                    preview_ref=preview.preview_ref,
                    authorization_ref=str(raw["authorization_ref"]),
                )
            )
        first = parts[0]
        if any(
            part.ordinal != ordinal
            or part.page_count != len(parts)
            or part.scope_kind != first.scope_kind
            or part.index_generation != first.index_generation
            or part.recipe_version != first.recipe_version
            or part.manifest_digest != first.manifest_digest
            or part.previous_preview_ref != (None if ordinal == 0 else parts[ordinal - 1].preview_ref)
            for ordinal, part in enumerate(parts)
        ):
            raise AuthorizationMismatchError("sealed insight manifest pages disagree on their immutable facts")
        if insight_manifest_digest(tuple(parts)) != first.manifest_digest:
            raise AuthorizationMismatchError("sealed insight manifest digest differs from its page facts")
        return tuple(parts)

    def machine_preview_summary(self, binding: MachineRequestBinding) -> dict[str, object]:
        """Describe a sealed preview from bounded durable facts, not a ref list."""
        from polylogue.operations.daemon_protocol import AcceptedOperationReference

        record = self.machine_request(binding)
        if record is None or record["artifact_kind"] != "preview-batch":
            raise ValueError("machine preview selection is not sealed")
        args = (binding.archive_identity, binding.request_id)
        with self._connection() as conn:
            facts = conn.execute(
                "SELECT COUNT(*), MIN(v.expires_at_ms) FROM machine_request_parts p "
                "JOIN operation_previews v ON v.preview_id = p.preview_ref "
                "WHERE p.archive_identity = ? AND p.request_id = ?",
                args,
            ).fetchone()
            count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM machine_request_parts p JOIN operation_preview_targets t "
                    "ON t.preview_id = p.preview_ref WHERE p.archive_identity = ? AND p.request_id = ?",
                    args,
                ).fetchone()[0]
            )
            sample = [
                str(row[0]).removeprefix("session:")
                for row in conn.execute(
                    "SELECT t.target_ref FROM machine_request_parts p JOIN operation_preview_targets t "
                    "ON t.preview_id = p.preview_ref WHERE p.archive_identity = ? AND p.request_id = ? "
                    "ORDER BY p.ordinal, t.ordinal LIMIT ?",
                    (*args, DELETE_PREVIEW_SAMPLE_IDS),
                )
            ]
            first = conn.execute(
                "SELECT preview_ref FROM machine_request_parts WHERE archive_identity = ? AND request_id = ? "
                "ORDER BY ordinal LIMIT 1",
                args,
            ).fetchone()
        if facts[0] != record["part_count"] or first is None:
            raise ValueError("machine preview authority is incomplete")
        return {
            "status": "prepared",
            "operation": "identity-reset" if binding.operation_name == "mutation.identity-reset.preview" else "delete",
            "reference": AcceptedOperationReference.from_record(record).to_dict(),
            **(
                {"preview_ref": str(first[0])}
                if facts[0] == 1 and binding.operation_name != "mutation.identity-reset.preview"
                else {}
            ),
            "session_ids_sample": sample,
            "session_count": count,
            **(
                {"lifetime": "accepted-request"}
                if binding.operation_name == "mutation.identity-reset.preview"
                else {"expires_at_ms": int(facts[1])}
            ),
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
        if getattr(self._settled_reader, "depth", 0):
            try:
                with open_verified_audit_read_connection(self.path) as conn:
                    conn.row_factory = sqlite3.Row
                    yield conn
            except AuditLeafError as exc:
                raise AuditContinuityPendingError("audit machine read is unavailable") from exc
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
            manifest, principal = (
                cast(FrozenSourceManifest | SealedSourceManifestRef, args[0]),
                cast(MutationPrincipal, args[1]),
            )
            if self._machine_binding is None or self._machine_binding[1] != kind:
                raise ValueError("ingest acceptance requires an authenticated machine binding")
            binding = self._machine_binding[0]
            if binding.principal_ref != principal.actor_ref or "archive.ingest" not in principal.capabilities:
                raise AuthorizationMismatchError("ingest acceptance principal lacks bound authority")
            if not isinstance(manifest, SealedSourceManifestRef):
                raise TypeError("new ingest acceptance requires a staged source manifest")
            plan = cast(MutationPlan | None, values.get("plan"))
            authorization = cast(MutationAuthorization | None, values.get("authorization"))
            if (plan is None) != (authorization is None):
                raise ValueError("ingest runtime authority requires both plan and authorization")
            payload: dict[str, object] = {"manifest": manifest.to_dict(), "principal": _principal_payload(principal)}
            if plan is None:
                assert authorization is None
                return payload
            assert authorization is not None
            validate_mutation_plan_integrity(plan)
            from polylogue.operations.ingest_acceptance import ingest_context, ingest_plan

            now_ms = int(time.time() * 1000)
            canonical_plan = ingest_plan(
                manifest,
                archive_instance_id=plan.archive_instance_id,
                archive_identity_digest=binding.archive_identity,
                now_ms=plan.prepared_at_ms,
                expires_at_ms=plan.expires_at_ms,
            )
            if plan != canonical_plan or dict(plan.context) != ingest_context(manifest):
                raise AuthorizationMismatchError("ingest runtime authority differs from the frozen source manifest")
            if (
                authorization.token is None
                or authorization.preview_ref != f"preview:{plan.plan_hash}"
                or authorization.authorization_id is not None
                or authorization.plan_hash != plan.plan_hash
                or authorization.actor != principal.actor_ref
                or authorization.role != (principal.role_label or "")
                or authorization.surface != principal.surface
                or authorization.confirmation_strength != "role_only"
                or authorization.capabilities != plan.required_capabilities
                or authorization.capability not in plan.required_capabilities
                or authorization.expires_at_ms != plan.expires_at_ms
                or now_ms >= plan.expires_at_ms
                or not set(plan.required_capabilities).issubset(principal.capabilities)
            ):
                raise AuthorizationMismatchError("ingest runtime authorization differs from the fresh exact plan")
            preview_id = f"preview:{secrets.token_urlsafe(18)}"
            authorization_id = f"authorization:{secrets.token_urlsafe(18)}"
            operation_id = f"operation:{secrets.token_urlsafe(18)}"
            attempt_id = f"attempt:{secrets.token_urlsafe(18)}"
            bound_authorization = replace(authorization, preview_ref=preview_id, authorization_id=authorization_id)
            payload.update(
                {
                    "plan": _replay_plan_payload(plan),
                    "preview_id": preview_id,
                    "authorization_id": authorization_id,
                    "operation_id": operation_id,
                    "attempt_id": attempt_id,
                    "issued_at_ms": now_ms,
                    "now_ms": now_ms,
                    "authorization": _authorization_payload(bound_authorization),
                    "authorization_token_sha256": token_sha256(authorization.token),
                    "ingest_operation_id": operation_id,
                }
            )
            return payload
        if kind in {"create_preview_batch", "issue_authorization_batch", "cancel_preview_batch"}:
            items, principal = cast(tuple[object, ...], args[0]), cast(MutationPrincipal, args[1])
            if not 1 <= len(items) <= 40:
                raise ValueError("authority batch must contain between 1 and 40 parts")
            child_kind = {
                "create_preview_batch": "create_preview",
                "issue_authorization_batch": "issue_authorization",
                "cancel_preview_batch": "cancel_preview",
            }[kind]
            commands: list[dict[str, object]] = []
            authorizations = cast(tuple[MutationAuthorization, ...], args[2]) if len(args) > 2 else ()
            as_of_ms = (
                self.handshake_as_of_ms(
                    self._machine_binding[0],
                    preview_refs=tuple(cast(MutationPreview, item).preview_ref for item in items),
                )
                if kind == "issue_authorization_batch" and self._machine_binding is not None
                else None
            )
            if authorizations and len(authorizations) != len(items):
                raise ValueError("authorization batch differs from preview batch")
            for ordinal, item in enumerate(items):
                child_args: tuple[object, ...]
                if child_kind == "cancel_preview":
                    child_args = (item,)
                elif child_kind == "create_preview":
                    child_args = (item, principal)
                else:
                    child_args = (item, principal, authorizations[ordinal])
                child_values: dict[str, object] = {}
                if child_kind == "issue_authorization" and self._identity_reset_intent is not None:
                    child_values["issued_at_ms"] = authorizations[ordinal].expires_at_ms
                child_payload = self._continuity_payload(child_kind, child_args, child_values)
                if child_kind == "issue_authorization" and as_of_ms is not None:
                    child_payload["authority_as_of_ms"] = as_of_ms
                commands.append(
                    AuditMutation(
                        kind=child_kind,
                        mutation_id=f"audit-part:{secrets.token_urlsafe(18)}",
                        created_at_ms=int(time.time() * 1000),
                        payload=child_payload,
                    ).command()
                )
            return {"commands": commands}
        if kind == "accept_execution_batch":
            refs, principal = cast(tuple[str, ...], args[0]), cast(MutationPrincipal, args[1])
            if not 1 <= len(refs) <= _MAX_MACHINE_AUTHORITY_PARTS or len(set(refs)) != len(refs):
                raise ValueError("execution batch must contain between 1 and 40 distinct authorizations")
            now_ms = int(time.time() * 1000)
            return {
                "authorization_refs": list(refs),
                "principal": _principal_payload(principal),
                "now_ms": now_ms,
                "authority_as_of_ms": (
                    self.handshake_as_of_ms(self._machine_binding[0], authorization_refs=refs)
                    if self._machine_binding is not None
                    else now_ms
                ),
            }
        if kind == "seal_insight_execution":
            head_preview_ref, page_count, manifest_digest, raw_principal = args
            principal = cast(MutationPrincipal, raw_principal)
            if (
                not isinstance(head_preview_ref, str)
                or type(page_count) is not int
                or not 1 <= page_count <= _MAX_INSIGHT_ACCEPTED_PARTS
                or not isinstance(manifest_digest, str)
                or len(manifest_digest) != 64
                or any(char not in "0123456789abcdef" for char in manifest_digest)
            ):
                raise ValueError("insight seal requires a bounded typed manifest head")
            if self._machine_binding is None or self._machine_binding[1] != kind:
                raise ValueError("insight seal requires an authenticated machine binding")
            binding = self._machine_binding[0]
            if binding.operation_name != _INSIGHT_MACHINE_OPERATION or binding.principal_ref != principal.actor_ref:
                raise AuthorizationMismatchError("insight seal principal or machine operation differs from authority")
            return {
                "head_preview_ref": head_preview_ref,
                "page_count": page_count,
                "manifest_digest": manifest_digest,
                "principal": _principal_payload(principal),
                "now_ms": int(time.time() * 1000),
            }
        if kind == "append_insight_preview":
            plan, principal = cast(MutationPlan, args[0]), cast(MutationPrincipal, args[1])
            if self._machine_binding is None or self._machine_binding[1] != kind:
                raise ValueError("insight preview staging requires an authenticated machine binding")
            binding = self._machine_binding[0]
            if binding.operation_name != _INSIGHT_MACHINE_OPERATION or binding.principal_ref != principal.actor_ref:
                raise AuthorizationMismatchError("insight preview staging principal or machine operation differs")
            from polylogue.operations.insight_acceptance import accepted_part_from_plan

            # Both the typed authority and its preview ref must be fixed before
            # source-WAL prepare.  Replaying a partial append cannot mint a
            # replacement predecessor for a later sealed page.
            accepted_part_from_plan(plan, preview_ref="pending-preview", authorization_ref="pending-authorization")
            if plan.archive_identity_digest != binding.archive_identity:
                raise AuthorizationMismatchError("insight preview archive differs from machine authority")
            return {
                "preview_id": f"preview:{secrets.token_urlsafe(18)}",
                "plan": _replay_plan_payload(plan),
                "principal": _principal_payload(principal),
            }
        if kind == "stop_machine_batch":
            reason = str(args[1])
            if reason not in {"cancelled", "deadline", "refused", "indeterminate", "interrupted"}:
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
            issued_at_ms = values.get("issued_at_ms")
            if issued_at_ms is None:
                issued_at_ms = int(time.time() * 1000)
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
                "issued_at_ms": cast(int, issued_at_ms),
                "preview": _preview_payload(preview),
                "principal": _principal_payload(principal),
                "authorization": _authorization_payload(authorization),
            }
        if kind == "consume_authorization_and_start":
            if isinstance(args[0], _StoredAuthorizationDigest):
                preview, authorization = cast(MutationPreview, args[1]), cast(MutationAuthorization, args[2])
            else:
                preview, authorization = cast(MutationPreview, args[0]), cast(MutationAuthorization, args[1])
            consume_now_ms = int(time.time() * 1000)
            return {
                "authority_as_of_ms": (
                    self.handshake_as_of_ms(
                        self._machine_binding[0],
                        authorization_refs=(authorization.authorization_id,),
                    )
                    if self._machine_binding is not None and authorization.authorization_id is not None
                    else consume_now_ms
                ),
                "operation_id": f"operation:{secrets.token_urlsafe(18)}",
                "attempt_id": f"attempt:{secrets.token_urlsafe(18)}",
                # The command can be replayed by a fresh repository process.
                # Keep the original actuator owner, rather than accidentally
                # assigning its pre-effect attempt to the recovery process.
                "attempt_owner_id": self._attempt_owner_id,
                "now_ms": consume_now_ms,
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
        if kind == "append_ingest_session_id_page":
            operation_id = cast(str, args[0])
            ordinal = cast(int, args[1])
            session_ids = cast(tuple[str, ...], args[2])
            if (
                not operation_id
                or type(ordinal) is not int
                or ordinal < 0
                or not 1 <= len(session_ids) <= MAX_PAGE_ITEMS
                or any(type(session_id) is not str or not session_id for session_id in session_ids)
                or list(session_ids) != sorted(set(session_ids))
            ):
                raise ValueError("ingest session ID page is not bounded, sorted, and unique")
            return {
                "operation_id": operation_id,
                "ordinal": ordinal,
                "session_ids": list(session_ids),
                "now_ms": int(time.time() * 1000),
            }
        if kind == "append_ingest_insight_page":
            operation_id = cast(str, args[0])
            page = cast(IngestInsightPageHistoricalReceipt, args[1])
            if not operation_id or not isinstance(page, IngestInsightPageHistoricalReceipt):
                raise ValueError("ingest insight page requires a typed operation and page")
            return {
                "operation_id": operation_id,
                "page": page.model_dump(mode="json"),
                "now_ms": int(time.time() * 1000),
            }
        if kind == "append_ingest_refusal_page":
            operation_id = cast(str, args[0])
            refusal_page = cast(IngestRefusalPageHistoricalReceipt, args[1])
            if not operation_id or not isinstance(refusal_page, IngestRefusalPageHistoricalReceipt):
                raise ValueError("ingest refusal page requires a typed operation and page")
            return {
                "operation_id": operation_id,
                "page": refusal_page.model_dump(mode="json"),
                "now_ms": int(time.time() * 1000),
            }
        if kind == "append_ingest_input_raw_page":
            operation_id = cast(str, args[0])
            raw_page = cast(IngestInputRawPageHistoricalReceipt, args[1])
            if not operation_id or not isinstance(raw_page, IngestInputRawPageHistoricalReceipt):
                raise ValueError("ingest input raw page requires a typed operation and page")
            return {
                "operation_id": operation_id,
                "page": raw_page.model_dump(mode="json"),
                "now_ms": int(time.time() * 1000),
            }
        if kind == "append_ingest_input_page":
            operation_id = cast(str, args[0])
            input_page = cast(IngestInputPageHistoricalReceipt, args[1])
            if not operation_id or not isinstance(input_page, IngestInputPageHistoricalReceipt):
                raise ValueError("ingest input page requires a typed operation and page")
            return {
                "operation_id": operation_id,
                "page": input_page.model_dump(mode="json"),
                "now_ms": int(time.time() * 1000),
            }
        if kind == "recover_abandoned_attempts":
            return {"now_ms": int(time.time() * 1000)}
        if kind == "resume_interrupted_ingest":
            return {
                "operation_id": cast(str, args[0]),
                "attempt_id": f"attempt:{secrets.token_urlsafe(18)}",
                "attempt_owner_id": self._attempt_owner_id,
                "now_ms": int(time.time() * 1000),
            }
        if kind == "record_recovery_resolution":
            resolver_actor_ref = kwargs["resolver_actor_ref"]
            if not isinstance(resolver_actor_ref, str) or not resolver_actor_ref:
                raise ValueError("recovery resolver actor_ref must not be empty")
            resolution = cast(RecoveryResolution, args[1])
            return {
                "operation_id": cast(str, args[0]),
                "resolver_actor_ref": resolver_actor_ref,
                "outcome": resolution.outcome,
                "detail": resolution.detail,
                "receipt": None if resolution.receipt is None else _receipt_payload(resolution.receipt),
                "now_ms": int(time.time() * 1000),
            }
        raise RuntimeError(f"unregistered audit continuity mutation {kind!r}")

    def _replay_pending_mutation(self, conn: sqlite3.Connection, mutation: AuditMutation) -> object:
        result = self._replay_domain_mutation(conn, mutation)
        self._bind_machine_result(conn, mutation, result)
        return result

    def _replay_domain_mutation(self, conn: sqlite3.Connection, mutation: AuditMutation) -> object:
        """Replay the stored typed command without allocating fresh ids or clocks."""

        payload = {} if mutation.kind == EXCISION_SOURCE_COMMIT_KIND else mutation.mapping_payload
        self._coordinated_connection = conn
        self._coordinated_mutation = mutation
        try:
            if mutation.kind == EXCISION_SOURCE_COMMIT_KIND:
                return self._project_excision_source_completion(conn, mutation)
            if mutation.kind in {"create_preview_batch", "issue_authorization_batch", "cancel_preview_batch"}:
                return self._apply_authority_batch()
            if mutation.kind == "accept_ingest":
                # The source-WAL prepare has already accepted this denominator.
                # Replaying the audit reference cannot reauthorize or acquire.
                manifest = source_manifest_from_dict(payload["manifest"])
                if "plan" not in payload:
                    return manifest.source_generation_id
                self._apply_ingest_runtime_payload(payload)
                return manifest.source_generation_id
            if mutation.kind == "append_insight_preview":
                return cast(Any, self.append_insight_preview).__wrapped__(
                    self,
                    _plan_from_payload(payload["plan"]),
                    _principal_from_payload(payload["principal"]),
                )
            if mutation.kind == "accept_execution_batch":
                return cast(Any, self.accept_execution_batch).__wrapped__(
                    self,
                    tuple(cast(list[str], payload["authorization_refs"])),
                    _principal_from_payload(payload["principal"]),
                )
            if mutation.kind == "seal_insight_execution":
                return cast(Any, self.seal_insight_execution).__wrapped__(
                    self,
                    cast(str, payload["head_preview_ref"]),
                    cast(int, payload["page_count"]),
                    cast(str, payload["manifest_digest"]),
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
            if mutation.kind == "append_ingest_session_id_page":
                return cast(Any, self.append_ingest_session_id_page).__wrapped__(
                    self,
                    cast(str, payload["operation_id"]),
                    cast(int, payload["ordinal"]),
                    tuple(cast(list[str], payload["session_ids"])),
                )
            if mutation.kind == "append_ingest_insight_page":
                return cast(Any, self.append_ingest_insight_page).__wrapped__(
                    self,
                    cast(str, payload["operation_id"]),
                    IngestInsightPageHistoricalReceipt.model_validate(payload["page"]),
                )
            if mutation.kind == "append_ingest_refusal_page":
                return cast(Any, self.append_ingest_refusal_page).__wrapped__(
                    self,
                    cast(str, payload["operation_id"]),
                    IngestRefusalPageHistoricalReceipt.model_validate(payload["page"]),
                )
            if mutation.kind == "append_ingest_input_raw_page":
                return cast(Any, self.append_ingest_input_raw_page).__wrapped__(
                    self,
                    cast(str, payload["operation_id"]),
                    IngestInputRawPageHistoricalReceipt.model_validate(payload["page"]),
                )
            if mutation.kind == "append_ingest_input_page":
                return cast(Any, self.append_ingest_input_page).__wrapped__(
                    self,
                    cast(str, payload["operation_id"]),
                    IngestInputPageHistoricalReceipt.model_validate(payload["page"]),
                )
            if mutation.kind == "recover_abandoned_attempts":
                return cast(Any, self._recover_abandoned_attempts).__wrapped__(self)
            if mutation.kind == "resume_interrupted_ingest":
                return cast(Any, self.resume_interrupted_ingest).__wrapped__(self, cast(str, payload["operation_id"]))
            if mutation.kind == "record_recovery_resolution":
                raw_receipt = payload.get("receipt")
                return cast(Any, self.record_recovery_resolution).__wrapped__(
                    self,
                    cast(str, payload["operation_id"]),
                    RecoveryResolution(
                        outcome=cast(Any, payload["outcome"]),
                        detail=cast(str, payload["detail"]),
                        receipt=None if raw_receipt is None else _receipt_from_payload(raw_receipt),
                    ),
                    resolver_actor_ref=cast(str, payload["resolver_actor_ref"]),
                )
            raise AuditContinuityUnknownMutationError(mutation.kind)
        finally:
            self._coordinated_mutation = None
            self._coordinated_connection = None

    @staticmethod
    def _project_excision_source_completion(conn: sqlite3.Connection, mutation: AuditMutation) -> None:
        """Bind the whole actual Source receipt to its original begun closure."""
        from polylogue.storage.io_phase_metrics import connection_cursor

        payload = mutation.payload
        if not isinstance(payload, CanonicalAuditLiteral):
            raise AuditContinuityError("excision Source completion lacks its native canonical receipt")
        operation_id, attempt_id, plan_hash = excision_completion_identity(payload)
        with connection_cursor(
            conn,
            "SELECT r.operation_name,r.plan_hash,r.status,a.state FROM operation_runs AS r "
            "JOIN operation_attempts AS a ON a.operation_id=r.operation_id "
            "WHERE r.operation_id=? AND a.attempt_id=? AND a.target_ordinal=0 "
            "AND a.rowid=(SELECT max(latest.rowid) FROM operation_attempts AS latest WHERE latest.operation_id=r.operation_id)",
            (operation_id, attempt_id),
        ) as cursor:
            original = cursor.fetchone()
        if (
            original is None
            or tuple(original[:2]) != ("mutate-session-excision", plan_hash)
            or tuple(original[2:]) not in {("running", "running"), ("interrupted", "unknown")}
        ):
            raise AuditContinuityError("excision Source completion differs from its current begun attempt")
        AuditRepository._append_event(
            conn,
            operation_id=operation_id,
            attempt_id=attempt_id,
            event_type="excision_source_committed",
            occurred_at_ms=mutation.created_at_ms,
            detail=payload,
        )
        # The event and continuity head share this transaction. Validate the
        # native JSON relation without hydrating any target or hash array.
        with connection_cursor(
            conn,
            "WITH receipt AS (SELECT e.detail_json FROM operation_events AS e "
            "WHERE e.operation_id=? AND e.attempt_id=? AND e.event_type='excision_source_committed'), "
            "actual AS (SELECT t.key AS ordinal,json_extract(t.value,'$.session_id') AS session_id,t.value AS detail "
            "FROM receipt,json_each(receipt.detail_json,'$.targets') AS t), "
            "expected AS (SELECT t.ordinal,t.target_ref,t.state,t.target_kind "
            "FROM operation_targets AS t WHERE t.operation_id=?), "
            "dispositions AS (SELECT h.value AS hash,1 AS removed FROM actual AS a,"
            "json_each(a.detail,'$.removed_blob_hashes') AS h UNION ALL "
            "SELECT h.value AS hash,0 AS removed FROM actual AS a,json_each(a.detail,'$.shared_blob_hashes') AS h) "
            "SELECT (SELECT count(*) FROM receipt)=1 "
            "AND (SELECT count(*) FROM actual)=(SELECT target_count FROM operation_runs WHERE operation_id=?) "
            "AND (SELECT count(*) FROM actual)=(SELECT count(DISTINCT session_id) FROM actual) "
            "AND NOT EXISTS(SELECT 1 FROM actual AS a LEFT JOIN expected AS e ON e.ordinal=a.ordinal "
            "WHERE e.ordinal IS NULL OR e.target_ref IS NOT ('session:'||a.session_id) OR a.session_id='' "
            "OR e.target_kind!='session' OR e.state!=CASE WHEN ? THEN 'unknown' "
            "WHEN e.ordinal=0 THEN 'running' ELSE 'pending' END) "
            "AND NOT EXISTS(SELECT 1 FROM expected AS e LEFT JOIN actual AS a ON a.ordinal=e.ordinal "
            "WHERE a.ordinal IS NULL) "
            "AND NOT EXISTS(SELECT 1 FROM dispositions GROUP BY hash HAVING min(removed)!=max(removed)) "
            "AND NOT EXISTS(SELECT 1 FROM operation_runs AS r JOIN operation_previews AS p ON p.preview_id=r.preview_id, "
            "json_each(p.plan_json,'$.replay_context.context.targets') AS frozen "
            "LEFT JOIN actual AS a ON a.session_id=json_extract(frozen.value,'$.session_id') "
            "WHERE r.operation_id=? AND a.ordinal IS NULL) "
            "AND (SELECT count(*) FROM operation_runs AS r JOIN operation_previews AS p ON p.preview_id=r.preview_id, "
            "json_each(p.plan_json,'$.replay_context.context.targets') WHERE r.operation_id=?)=(SELECT count(*) FROM actual)",
            (
                operation_id,
                attempt_id,
                operation_id,
                operation_id,
                original[2] == "interrupted",
                operation_id,
                operation_id,
            ),
        ) as cursor:
            complete = cursor.fetchone()
        if complete is None or complete[0] != 1:
            raise AuditContinuityError("excision Source receipts differ from the complete original target closure")

    def _command_value(self, key: str, fallback: object) -> object:
        if self._coordinated_mutation is None:
            return fallback
        return self._coordinated_mutation.mapping_payload.get(key, fallback)

    def _begin(self, conn: sqlite3.Connection) -> None:
        """Start a standalone audit transaction, or reuse the coordinator's one."""

        if self._coordinated_connection is None:
            conn.execute("BEGIN IMMEDIATE")

    def _apply_authority_batch(self) -> list[str]:
        outer, conn = self._coordinated_mutation, self._coordinated_connection
        if outer is None or conn is None:
            raise RuntimeError("authority batches require source-WAL continuity")
        results: list[str] = []
        previous_proof = self._accepted_reset_custody
        commands = cast(list[dict[str, object]], outer.mapping_payload["commands"])
        raw_binding = outer.mapping_payload.get("machine_request")
        reset = (
            isinstance(raw_binding, dict) and raw_binding.get("operation_name") == "mutation.identity-reset.authorize"
        )
        try:
            for index, command in enumerate(commands):
                child = AuditMutation.from_command(command)
                self._accepted_reset_custody = None
                if reset:
                    page = cast(dict[str, object], outer.mapping_payload["machine_page"])
                    payload = child.mapping_payload
                    if outer.kind != "issue_authorization_batch" or child.kind != "issue_authorization":
                        raise AuthorizationMismatchError("reset custody belongs to another authority transition")
                    proof = self._verify_identity_reset_custody(
                        conn,
                        command=outer.mapping_payload,
                        transition=outer.kind,
                        ordinal=cast(int, page["offset"]) + index,
                        preview=_preview_from_payload(payload["preview"]),
                        principal=_principal_from_payload(payload["principal"]),
                        issued_at_ms=cast(int, payload["issued_at_ms"]),
                    )
                    self._require_reset_page_completion(page, len(commands), proof)
                    self._accepted_reset_custody = proof
                result = self._replay_domain_mutation(conn, child)
                if outer.kind == "cancel_preview_batch":
                    result = cast(dict[str, object], cast(dict[str, object], command["payload"])["preview"])[
                        "preview_ref"
                    ]
                if not isinstance(result, str):
                    raise TokenExpiredError("authority batch includes an expired preview")
                results.append(result)
            return results
        finally:
            self._accepted_reset_custody = previous_proof
            self._coordinated_mutation, self._coordinated_connection = outer, conn

    @staticmethod
    def _require_reset_page_completion(
        page: Mapping[str, object], count: int, proof: AcceptedIdentityResetCustody
    ) -> None:
        end = cast(int, page["offset"]) + count
        if end > proof.source_part_count or page["final"] != (end == proof.source_part_count):
            raise AuthorizationMismatchError("reset page does not close its exact source population")

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
    def accept_ingest(
        self,
        manifest: SealedSourceManifestRef,
        principal: MutationPrincipal,
        *,
        plan: MutationPlan | None = None,
        authorization: MutationAuthorization | None = None,
    ) -> str:
        """Bind retained physical inputs; source preparation owns acceptance."""
        if not isinstance(manifest, SealedSourceManifestRef):
            raise TypeError("new ingest acceptance requires a staged source manifest")
        if plan is not None:
            if self._coordinated_mutation is None:
                raise RuntimeError("paired ingest authority requires source-WAL coordination")
            self._apply_ingest_runtime_payload(self._coordinated_mutation.mapping_payload)
        return manifest.source_generation_id

    def _apply_ingest_runtime_payload(self, payload: Mapping[str, object]) -> None:
        """Apply the one frozen paired-ingest authority in normal and replay paths."""

        plan = _plan_from_payload(payload["plan"])
        preview = MutationPreview(cast(str, payload["preview_id"]), plan)
        principal = _principal_from_payload(payload["principal"])
        authorization = _authorization_from_payload(payload["authorization"])
        digest = _StoredAuthorizationDigest(cast(str, payload["authorization_token_sha256"]))
        self._create_preview_without_continuity(plan, principal)
        self._persist_authorization(
            digest, preview, principal, authorization, issued_at_ms=cast(int, payload["issued_at_ms"])
        )
        self._consume_authorization(digest, preview, authorization)

    @_continuity_mutation("accept_execution_batch")
    def accept_execution_batch(self, refs: tuple[str, ...], principal: MutationPrincipal) -> list[str]:
        """Reserve ordered authority; a domain run is created only before its effect."""
        self._validate_execution_reservations(refs, principal)
        return list(refs)

    def _validate_execution_reservations(self, refs: tuple[str, ...], principal: MutationPrincipal) -> None:
        """Check exact one-shot authority without allocating a domain attempt."""

        now_ms = int(cast(int, self._command_value("now_ms", int(time.time() * 1000))))
        now_ms = int(cast(int, self._command_value("authority_as_of_ms", now_ms)))
        with self._connection() as conn:
            for index, ref in enumerate(refs):
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
                command = {} if self._coordinated_mutation is None else self._coordinated_mutation.mapping_payload
                raw_binding = command.get("machine_request")
                reset = isinstance(raw_binding, dict) and raw_binding.get("operation_name") == "mutation.identity-reset"
                if reset:
                    preview, _ = self.authorization_for_principal(ref, principal)
                    page = cast(dict[str, object], command["machine_page"])
                    proof = self._verify_identity_reset_custody(
                        conn,
                        command=command,
                        transition="accept_execution_batch",
                        ordinal=cast(int, page["offset"]) + index,
                        preview=preview,
                        principal=principal,
                        issued_at_ms=now_ms,
                        authorization_ref=ref,
                    )
                    self._require_reset_page_completion(page, len(refs), proof)
                elif int(row[3]) <= now_ms:
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

    @_continuity_mutation("seal_insight_execution")
    def seal_insight_execution(
        self,
        head_preview_ref: str,
        page_count: int,
        manifest_digest: str,
        principal: MutationPrincipal,
    ) -> list[str]:
        """Seal a typed insights page chain into existing exact machine parts.

        The WAL command contains only the manifest head/count/digest.  Each
        page's immutable preview context provides its predecessor, exact
        target sequence, generation and recipe facts; no post-accept archive
        discovery or unbounded serialized reference list is available here.
        """

        parts = self._resolve_insight_manifest(
            head_preview_ref,
            page_count,
            manifest_digest,
            principal,
            expected_archive_identity=self._insight_bound_archive_identity(),
        )
        refs = tuple(part.authorization_ref for part in parts)
        self._validate_execution_reservations(refs, principal)
        return list(refs)

    def _resolve_insight_manifest(
        self,
        head_preview_ref: str,
        page_count: int,
        manifest_digest: str,
        principal: MutationPrincipal,
        *,
        expected_archive_identity: str,
    ) -> tuple[AcceptedInsightPart, ...]:
        """Reconstruct and verify an ordered bounded preview chain from audit rows."""

        from polylogue.operations.insight_acceptance import (
            MAX_INSIGHT_ACCEPTED_PARTS,
            accepted_part_from_plan,
            insight_manifest_digest,
        )

        if not 1 <= page_count <= MAX_INSIGHT_ACCEPTED_PARTS:
            raise ValueError("insight manifest page count exceeds the bounded authority budget")
        cursor = head_preview_ref
        reverse: list[AcceptedInsightPart] = []
        seen: set[str] = set()
        with self._connection() as conn:
            for expected_ordinal in range(page_count - 1, -1, -1):
                if cursor in seen:
                    raise AuthorizationMismatchError("insight manifest preview chain contains a cycle")
                seen.add(cursor)
                preview = self.preview_for_principal(cursor, principal)
                if preview.plan.operation != "mutate-rebuild-insights":
                    raise AuthorizationMismatchError("insight manifest preview has another operation")
                if preview.plan.archive_identity_digest != expected_archive_identity:
                    raise AuthorizationMismatchError("insight manifest preview has another archive authority")
                rows = conn.execute(
                    """SELECT authorization_id FROM operation_authorizations
                    WHERE preview_id = ? AND actor_ref = ? AND surface = ? AND state = 'active'
                    ORDER BY authorization_id""",
                    (cursor, principal.actor_ref, principal.surface),
                ).fetchall()
                if len(rows) != 1:
                    raise AuthorizationMismatchError("insight manifest page lacks one exact active authorization")
                part = accepted_part_from_plan(preview.plan, preview_ref=cursor, authorization_ref=str(rows[0][0]))
                if part.ordinal != expected_ordinal or part.page_count != page_count:
                    raise AuthorizationMismatchError("insight manifest page ordering differs from its sealed count")
                reverse.append(part)
                cursor = part.previous_preview_ref or ""
        if cursor:
            raise AuthorizationMismatchError("insight manifest chain has an unexpected predecessor")
        parts = tuple(reversed(reverse))
        first = parts[0]
        if any(
            part.scope_kind != first.scope_kind
            or part.index_generation != first.index_generation
            or part.recipe_version != first.recipe_version
            or part.manifest_digest != manifest_digest
            for part in parts
        ):
            raise AuthorizationMismatchError("insight manifest pages disagree on immutable acceptance facts")
        if insight_manifest_digest(parts) != manifest_digest:
            raise AuthorizationMismatchError("insight manifest digest does not match its exact page chain")
        return parts

    def _insight_bound_archive_identity(self) -> str:
        """Resolve the authenticated request archive in live and replayed seals."""

        if self._machine_binding is not None:
            binding, transition = self._machine_binding
            if transition == "seal_insight_execution" and binding.operation_name == _INSIGHT_MACHINE_OPERATION:
                return binding.archive_identity
        if self._coordinated_mutation is not None:
            raw = self._coordinated_mutation.mapping_payload.get("machine_request")
            if isinstance(raw, dict) and raw.get("operation_name") == _INSIGHT_MACHINE_OPERATION:
                archive_identity = raw.get("archive_identity")
                if isinstance(archive_identity, str):
                    return archive_identity
        raise AuthorizationMismatchError("insight seal has no authenticated machine archive binding")

    @_continuity_mutation("cancel_preview_batch")
    def cancel_preview_batch(self, previews: tuple[MutationPreview, ...], principal: MutationPrincipal) -> list[str]:
        """Cancel an exact ordered preview set without consuming its authority."""
        return self._apply_authority_batch()

    def fence_staged_machine_pages(self) -> int:
        """Stop every paged batch a dead daemon left half-accepted.

        Runs at the single-writer startup seam, where no handler can still be
        appending pages. The request reads as interrupted, so a caller
        following it reaches a terminal outcome and submits it again.
        """
        placeholders = ",".join("?" for _ in MACHINE_PAGE_KINDS)
        with self._connection() as conn:
            rows = conn.execute(
                f"""SELECT archive_identity, request_id, principal_ref, fingerprint, operation_name
                    FROM machine_requests WHERE stop_reason IS NULL AND artifact_kind IN ({placeholders})""",
                tuple(sorted(MACHINE_PAGE_KINDS)),
            ).fetchall()
        for row in rows:
            self.stop_machine_batch(MachineRequestBinding(*(str(value) for value in row)), "interrupted")
        return len(rows)

    @_continuity_mutation("stop_machine_batch")
    def stop_machine_batch(self, binding: MachineRequestBinding, reason: str) -> None:
        """Fence unstarted parts without hiding or revoking an attempted effect.

        A staged authorization batch has attempted nothing, so its issued
        authorizations are revoked with it.
        """
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
            # A staged authorization batch keeps its issued authorizations as
            # its parts' artifacts. Fencing it must not leave the pages it
            # already accepted usable by an execution nobody can follow.
            conn.execute(
                """UPDATE operation_authorizations SET state = 'revoked'
                WHERE state = 'active' AND authorization_id IN (
                    SELECT parts.artifact_ref FROM machine_request_parts AS parts
                    JOIN machine_requests AS requests
                      ON requests.archive_identity = parts.archive_identity
                     AND requests.request_id = parts.request_id
                    WHERE parts.archive_identity = ? AND parts.request_id = ?
                      AND requests.artifact_kind = ?
                )""",
                (binding.archive_identity, binding.request_id, machine_pages_kind("authorization-batch")),
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

    @_continuity_mutation("append_insight_preview")
    def append_insight_preview(self, plan: MutationPlan, principal: MutationPrincipal) -> str:
        """Stage exactly one closed insight page before it is accepted for execution.

        A staged preview is distinguishable from an execution acceptance in
        ``machine_requests.artifact_kind``.  The same request can therefore
        resume its page chain without silently minting a replacement preview.
        """

        return self._create_preview_without_continuity(plan, principal)

    def _create_preview_without_continuity(self, plan: MutationPlan, principal: MutationPrincipal) -> str:
        """Reuse preview persistence while the outer continuity command is active."""

        raw = getattr(self.create_preview, "__wrapped__", None)
        if not callable(raw):
            raise RuntimeError("create preview lost its continuity implementation")
        create_preview = cast(Callable[[AuditRepository, MutationPlan, MutationPrincipal], str], raw)
        return create_preview(self, plan, principal)

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
            proof = self._accepted_reset_custody
            if proof is not None and (
                proof.preview_ref != preview.preview_ref
                or proof.plan_hash != preview.plan.plan_hash
                or proof.principal != principal
                or proof.issued_at_ms != effective_issued_at_ms
                or authorization.expires_at_ms != effective_issued_at_ms
            ):
                raise AuthorizationMismatchError("authorization differs from its verified reset custody")
            durable_expires_at_ms = int(preview_row[1]) if proof is None else effective_issued_at_ms
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
            # A paged handshake judges expiry as of its progress, not its wall
            # time (``handshake_as_of_ms``); issuance time stays real.
            as_of_ms = cast(int, self._command_value("authority_as_of_ms", effective_issued_at_ms))
            if proof is None and as_of_ms >= durable_expires_at_ms:
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
            command = {} if self._coordinated_mutation is None else self._coordinated_mutation.mapping_payload
            if preview.plan.operation == "mutate-rebuild-insights" and reservation is None:
                raise TokenConsumedError("unsealed insight authority cannot start an execution")
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
            raw_binding = command.get("machine_request")
            reset = isinstance(raw_binding, dict) and raw_binding.get("operation_name") == "mutation.identity-reset"
            if reset:
                principal = MutationPrincipal(
                    actor_ref=authorization.actor,
                    surface=cast(Any, authorization.surface),
                    role_label=authorization.role or None,
                    capabilities=frozenset(authorization.capabilities),
                )
                self._verify_identity_reset_custody(
                    conn,
                    command=command,
                    transition="consume_authorization_and_start",
                    ordinal=cast(int, command["machine_part"]),
                    preview=preview,
                    principal=principal,
                    issued_at_ms=now_ms,
                    authorization_ref=str(row[0]),
                )
            if not reset and int(row[7]) <= cast(int, self._command_value("authority_as_of_ms", now_ms)):
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
                history = receipt.historical_receipt
                if isinstance(history, IngestHistoricalReceiptV2):
                    if history.input_pages_ref != operation_id:
                        raise ValueError("historical ingest input pages belong to another operation")
                    for _page in self._scan_ingest_input_pages(conn, history):
                        pass
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
        with self._connection() as conn:
            page_size = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
            rows_by_id: dict[str, sqlite3.Row] = {}
            for offset in range(0, len(target_refs), page_size):
                page = target_refs[offset : offset + page_size]
                placeholders = ", ".join("?" for _ in page)
                rows = conn.execute(
                    f"""
                    SELECT DISTINCT r.operation_id, r.operation_name, r.operation_version,
                           r.plan_hash, r.target_digest, r.started_at_ms
                    FROM operation_runs AS r
                    JOIN operation_targets AS t ON t.operation_id = r.operation_id
                    WHERE r.status IN ('running', 'interrupted')
                      AND t.target_ref IN ({placeholders})
                    """,
                    page,
                ).fetchall()
                for row in rows:
                    rows_by_id[str(row["operation_id"])] = row
            ordered = sorted(rows_by_id.values(), key=lambda row: (row["started_at_ms"], row["operation_id"]))
            operations = [self._recovery_operation(conn, row) for row in ordered]
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
        attempt = conn.execute(
            "SELECT attempt_id FROM operation_attempts WHERE operation_id=? ORDER BY rowid DESC LIMIT 1",
            (operation_id,),
        ).fetchone()
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
            attempt_id=None if attempt is None else str(attempt[0]),
        )

    def operation_plan(self, operation_id: str) -> MutationPlan:
        """Return the exact plan an operation was authorized and started with."""

        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT p.plan_json FROM operation_previews AS p
                JOIN operation_runs AS r ON r.preview_id = p.preview_id
                WHERE r.operation_id = ?
                """,
                (operation_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"operation {operation_id!r} has no recorded plan")
        return plan_from_stored_payload(json.loads(str(row[0])))

    @_continuity_mutation("record_recovery_resolution")
    def record_recovery_resolution(
        self, operation_id: str, resolution: RecoveryResolution, *, resolver_actor_ref: str
    ) -> None:
        """Terminalize one dead operation with the outcome its actuator decided.

        ``complete`` records the replay receipt's per-target outcome and
        completes the run; ``absent``, ``not-replayable`` and ``replay-failed``
        fail it. Every outcome is terminal, so no later request overlapping
        these targets meets it again. The event names the resolver; the run
        retains its original mutation actor.
        """

        if not resolver_actor_ref:
            raise ValueError("recovery resolver actor_ref must not be empty")
        now_ms = cast(int, self._command_value("now_ms", int(time.time() * 1000)))
        with self._connection() as conn:
            self._begin(conn)
            run = conn.execute(
                "SELECT status, plan_hash FROM operation_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if run is None:
                raise ValueError(f"unknown operation {operation_id!r}")
            status = str(run[0])
            if status not in {"running", "interrupted"}:
                return
            live_owner = conn.execute(
                "SELECT worker_id FROM operation_attempts WHERE operation_id = ? AND state = 'running'", (operation_id,)
            ).fetchall()
            owner_liveness = {_attempt_owner_liveness(cast(str | None, row[0])) for row in live_owner}
            # A live or unverifiable owner may still be mutating its targets.
            if owner_liveness & {"live", "unknown"}:
                return
            receipt = resolution.receipt
            if receipt is not None and receipt.plan_hash != str(run[1]):
                raise ValueError("recovery receipt does not match the interrupted plan")
            if resolution.outcome == "complete":
                assert receipt is not None
                target_state = "already_satisfied" if receipt.status == "already_satisfied" else "applied"
                run_state, reason = "completed", "recovered_complete"
            elif resolution.outcome == "absent":
                target_state, run_state, reason = "failed", "failed", "recovered_absent"
            elif resolution.outcome == "not-replayable":
                target_state, run_state, reason = "failed", "failed", "recovery_not_replayable"
            else:
                target_state, run_state, reason = "failed", "failed", "recovery_replay_failed"
            detail = resolution.detail[:512]
            conn.execute(
                """
                UPDATE operation_attempts SET state = 'reconciled', finished_at_ms = ?, unknown_reason = NULL,
                    error_summary = ?
                WHERE operation_id = ? AND state IN ('running', 'unknown')
                """,
                (now_ms, detail, operation_id),
            )
            conn.execute(
                """
                UPDATE operation_targets
                SET state = ?, completed_at_ms = ?, unknown_reason = NULL, error_summary = ?,
                    domain_receipt_ref = ?, domain_receipt_kind = ?
                WHERE operation_id = ? AND state IN ('pending', 'running', 'unknown')
                """,
                (
                    target_state,
                    now_ms,
                    None if run_state == "completed" else detail,
                    f"mutation-operation:{operation_id}" if receipt is not None else None,
                    "recovery-replay" if receipt is not None else None,
                    operation_id,
                ),
            )
            conn.execute(
                """
                UPDATE operation_runs
                SET status = ?, terminal_reason = ?, updated_at_ms = ?, completed_at_ms = ?,
                    unknown_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'unknown'),
                    affected_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'applied'),
                    failed_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'failed'),
                    unknown_reason = NULL, error_summary = ?
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
                    None if run_state == "completed" else detail,
                    operation_id,
                ),
            )
            self._append_event(
                conn,
                operation_id=operation_id,
                event_type="recovery_resolved",
                from_state=status,
                to_state=run_state,
                actor_ref=resolver_actor_ref,
                occurred_at_ms=now_ms,
                detail={
                    "outcome": resolution.outcome,
                    "detail": detail,
                    "affected_count": 0 if receipt is None else receipt.affected_count,
                    **(
                        {"historical_receipt": encode_machine_receipt(receipt.historical_receipt)}
                        if receipt is not None and receipt.historical_receipt is not None
                        else {}
                    ),
                },
            )

    def accepted_ingest_stop_reason(self, source_generation_id: str) -> tuple[bool, str | None]:
        """Return whether a machine request accepted this generation, and its stop reason."""

        with self._connection() as conn:
            row = conn.execute(
                "SELECT stop_reason FROM machine_requests WHERE artifact_kind = 'source-generation' "
                "AND artifact_ref = ? ORDER BY accepted_at_ms LIMIT 1",
                (source_generation_id,),
            ).fetchone()
        if row is None:
            return False, None
        return True, None if row[0] is None else str(row[0])

    def interrupted_ingest_requests(self) -> tuple[tuple[str, dict[str, object]], ...]:
        """Accepted, unstopped ingests whose run has no terminal checkpoint and no live owner.

        Each item is the operation id and its accepted machine request record.
        A running attempt whose owner is live or unverifiable still owns its
        effect and is not returned.
        """

        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT r.operation_id, m.*
                FROM operation_runs AS r
                JOIN machine_request_parts AS p ON p.operation_id = r.operation_id
                JOIN machine_requests AS m
                  ON m.archive_identity = p.archive_identity AND m.request_id = p.request_id
                WHERE r.operation_name = ? AND r.status IN ('running', 'interrupted')
                  AND m.artifact_kind = 'source-generation' AND m.stop_reason IS NULL
                ORDER BY m.accepted_at_ms, r.operation_id
                """,
                (INGEST_OPERATION,),
            ).fetchall()
            candidates: list[tuple[str, dict[str, object]]] = []
            for row in rows:
                record = dict(row)
                operation_id = str(record.pop("operation_id"))
                owners = conn.execute(
                    "SELECT worker_id FROM operation_attempts WHERE operation_id = ? AND state = 'running'",
                    (operation_id,),
                ).fetchall()
                if all(_attempt_owner_liveness(cast(str | None, owner[0])) == "dead" for owner in owners):
                    candidates.append((operation_id, record))
        return tuple(candidates)

    def ingest_operation_authority(self, operation_id: str) -> tuple[MutationPlan, MutationAuthorization]:
        """Return the exact plan and consumed authorization an accepted ingest started with."""

        with self._connection() as conn:
            run = conn.execute(
                "SELECT operation_name, initial_authorization_id FROM operation_runs WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if run is None or str(run[0]) != INGEST_OPERATION:
                raise ValueError(f"operation {operation_id!r} is not an accepted ingest")
            authorization_ref = str(run[1])
            row = conn.execute(
                "SELECT * FROM operation_authorizations WHERE authorization_id = ?", (authorization_ref,)
            ).fetchone()
            if row is None:
                raise ValueError("accepted ingest authorization is missing")
            record = dict(row)
            capabilities = tuple(
                str(item[0])
                for item in conn.execute(
                    "SELECT capability FROM operation_authorization_capabilities WHERE authorization_id = ? "
                    "ORDER BY capability",
                    (authorization_ref,),
                )
            )
        plan = self.operation_plan(operation_id)
        authorization = MutationAuthorization(
            plan_hash=plan.plan_hash,
            actor=str(record["actor_ref"]),
            role=str(record["role_label"] or ""),
            capability=capabilities[0] if capabilities else "",
            confirmation_strength=cast(Any, record["confirmation_strength"]),
            authorized_at=str(record["issued_at_ms"]),
            preview_ref=str(record["preview_id"]),
            authorization_id=authorization_ref,
            token=None,
            expires_at_ms=int(cast(int, record["expires_at_ms"])),
            capabilities=capabilities,
            surface=cast(Any, record["surface"]),
        )
        return plan, authorization

    @_continuity_mutation("resume_interrupted_ingest")
    def resume_interrupted_ingest(self, operation_id: str) -> bool:
        """Start a new attempt on an interrupted accepted ingest owned by this process.

        Succeeds only for an ingest run without a terminal checkpoint whose
        accepted request was never stopped and whose earlier attempts have no
        live or unverifiable owner. The run returns to ``running`` under the
        new attempt, so exactly one owner re-drives it and its request reads
        ``running`` until that owner finalizes it. Returns whether the claim
        was taken.
        """

        now_ms = cast(int, self._command_value("now_ms", int(time.time() * 1000)))
        attempt_id = cast(str, self._command_value("attempt_id", f"attempt:{secrets.token_urlsafe(18)}"))
        owner_id = cast(str | None, self._command_value("attempt_owner_id", self._attempt_owner_id))
        with self._connection() as conn:
            self._begin(conn)
            run = conn.execute(
                "SELECT operation_name, status, actor_ref, initial_authorization_id FROM operation_runs "
                "WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if run is None or str(run[0]) != INGEST_OPERATION or str(run[1]) not in {"running", "interrupted"}:
                return False
            request = conn.execute(
                """SELECT m.stop_reason FROM machine_request_parts AS p
                JOIN machine_requests AS m ON m.archive_identity = p.archive_identity AND m.request_id = p.request_id
                WHERE p.operation_id = ? AND m.artifact_kind = 'source-generation'""",
                (operation_id,),
            ).fetchone()
            if request is None or request[0] is not None:
                return False
            owners = conn.execute(
                "SELECT worker_id FROM operation_attempts WHERE operation_id = ? AND state = 'running'",
                (operation_id,),
            ).fetchall()
            if any(_attempt_owner_liveness(cast(str | None, owner[0])) != "dead" for owner in owners):
                return False
            conn.execute(
                "UPDATE operation_attempts SET state = 'unknown', finished_at_ms = ?, unknown_reason = ? "
                "WHERE operation_id = ? AND state = 'running'",
                (now_ms, "process ended before audit finalization", operation_id),
            )
            conn.execute(
                """
                UPDATE operation_targets
                SET state = 'running', attempt_count = attempt_count + 1, current_attempt_id = ?,
                    completed_at_ms = NULL, error_summary = NULL, unknown_reason = NULL
                WHERE operation_id = ? AND state IN ('pending', 'running', 'unknown')
                """,
                (attempt_id, operation_id),
            )
            conn.execute(
                """
                INSERT INTO operation_attempts(
                    attempt_id, operation_id, target_ordinal, authorization_id, worker_id, state, started_at_ms
                ) VALUES (?, ?, 0, ?, ?, 'running', ?)
                """,
                (attempt_id, operation_id, str(run[3]), owner_id, now_ms),
            )
            conn.execute(
                """
                UPDATE operation_runs
                SET status = 'running', terminal_reason = NULL, updated_at_ms = ?, completed_at_ms = NULL,
                    unknown_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'unknown'),
                    unknown_reason = NULL, error_summary = NULL
                WHERE operation_id = ?
                """,
                (now_ms, operation_id, operation_id),
            )
            self._append_event(
                conn,
                operation_id=operation_id,
                target_ordinal=0,
                attempt_id=attempt_id,
                event_type="attempt_resumed",
                from_state=str(run[1]),
                to_state="running",
                actor_ref=str(run[2]),
                occurred_at_ms=now_ms,
                detail={"attempt_id": attempt_id, "reason": "accepted generation re-driven by its ingest owner"},
            )
        return True

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

    def list_events(self, operation_id: str) -> tuple[dict[str, object], ...]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM operation_events WHERE operation_id = ? ORDER BY sequence", (operation_id,)
            ).fetchall()
            return tuple(dict(row) for row in rows)

    def last_event_sequence(self, operation_id: str) -> int:
        """Read lifecycle position without materializing historical page payloads."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT MAX(sequence) FROM operation_events WHERE operation_id=?", (operation_id,)
            ).fetchone()
            if row is None or row[0] is None:
                return 0
            if type(row[0]) is not int or row[0] < 0:
                raise ValueError("audit event sequence is malformed")
            return row[0]

    @_continuity_mutation("append_ingest_session_id_page")
    def append_ingest_session_id_page(self, operation_id: str, ordinal: int, session_ids: tuple[str, ...]) -> None:
        """Retain one bounded changed-ID page before the terminal receipt refers to it."""
        if (
            type(ordinal) is not int
            or ordinal < 0
            or not 1 <= len(session_ids) <= MAX_PAGE_ITEMS
            or any(type(session_id) is not str or not session_id for session_id in session_ids)
            or list(session_ids) != sorted(set(session_ids))
        ):
            raise ValueError("ingest session ID page is not bounded, sorted, and unique")
        with self._connection() as conn:
            run = conn.execute(
                "SELECT operation_name, status FROM operation_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if run is None or str(run[0]) != INGEST_OPERATION or str(run[1]) == "completed":
                raise ValueError("ingest session ID page lacks an open ingest operation")
            last_row = conn.execute(
                "SELECT detail_json FROM operation_events WHERE operation_id = ? AND event_type = 'ingest_session_id_page' "
                "ORDER BY sequence DESC LIMIT 1",
                (operation_id,),
            ).fetchone()
            last_page = None if last_row is None else json.loads(str(last_row[0]))
            last_ordinal = -1 if last_page is None else int(last_page["ordinal"])
            if ordinal <= last_ordinal:
                prior_row = conn.execute(
                    "SELECT detail_json FROM operation_events WHERE operation_id = ? "
                    "AND event_type = 'ingest_session_id_page' AND json_extract(detail_json, '$.ordinal') = ?",
                    (operation_id, ordinal),
                ).fetchone()
                prior = None if prior_row is None else json.loads(str(prior_row[0]))
                if prior != {"ordinal": ordinal, "session_ids": list(session_ids)}:
                    raise ValueError("ingest session ID page conflicts with durable page")
                return
            if ordinal != last_ordinal + 1:
                raise ValueError("ingest session ID pages must be contiguous")
            if last_page is not None and str(last_page["session_ids"][-1]) >= session_ids[0]:
                raise ValueError("ingest session ID pages must be globally sorted")
            self._append_event(
                conn,
                operation_id=operation_id,
                event_type="ingest_session_id_page",
                occurred_at_ms=cast(int, self._command_value("now_ms", int(time.time() * 1000))),
                detail={"ordinal": ordinal, "session_ids": list(session_ids)},
            )

    def read_ingest_session_id_pages(
        self, operation_id: str, *, page_count: int, session_count: int, digest: str
    ) -> list[str]:
        """Resolve immutable pages with exact count, order, and digest checks."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT detail_json FROM operation_events WHERE operation_id = ? AND event_type = 'ingest_session_id_page' "
                "ORDER BY sequence",
                (operation_id,),
            ).fetchall()
        if len(rows) != page_count:
            raise ValueError("ingest session ID page count differs from terminal receipt")
        session_ids: list[str] = []
        for ordinal, row in enumerate(rows):
            page = json.loads(str(row[0]))
            if (
                not isinstance(page, dict)
                or page.get("ordinal") != ordinal
                or not isinstance(page.get("session_ids"), list)
                or not 1 <= len(page["session_ids"]) <= MAX_PAGE_ITEMS
            ):
                raise ValueError("ingest session ID page is malformed or out of order")
            session_ids.extend(page["session_ids"])
        if (
            len(session_ids) != session_count
            or session_ids != sorted(set(session_ids))
            or ingest_session_ids_digest(session_ids) != digest
        ):
            raise ValueError("ingest session ID pages differ from terminal receipt")
        return session_ids

    @_continuity_mutation("append_ingest_insight_page")
    def append_ingest_insight_page(self, operation_id: str, page: IngestInsightPageHistoricalReceipt) -> None:
        """Persist one bounded profile-target page through audit continuity."""
        with self._connection() as conn:
            run = conn.execute(
                "SELECT operation_name, status FROM operation_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if run is None or str(run[0]) != INGEST_OPERATION or str(run[1]) == "completed":
                raise ValueError("ingest insight page lacks an open ingest operation")
            last_row = conn.execute(
                "SELECT detail_json FROM operation_events WHERE operation_id = ? AND event_type = 'ingest_insight_page' "
                "ORDER BY sequence DESC LIMIT 1",
                (operation_id,),
            ).fetchone()
            last_page = None if last_row is None else json.loads(str(last_row[0]))
            last_ordinal = -1 if last_page is None else int(last_page["ordinal"])
            payload = page.model_dump(mode="json")
            if page.ordinal <= last_ordinal:
                prior_row = conn.execute(
                    "SELECT detail_json FROM operation_events WHERE operation_id = ? "
                    "AND event_type = 'ingest_insight_page' AND json_extract(detail_json, '$.ordinal') = ?",
                    (operation_id, page.ordinal),
                ).fetchone()
                prior = None if prior_row is None else json.loads(str(prior_row[0]))
                if prior != payload:
                    raise ValueError("ingest insight page conflicts with durable page")
                return
            if page.ordinal != last_ordinal + 1:
                raise ValueError("ingest insight pages must be contiguous")
            self._append_event(
                conn,
                operation_id=operation_id,
                event_type="ingest_insight_page",
                occurred_at_ms=cast(int, self._command_value("now_ms", int(time.time() * 1000))),
                detail=payload,
            )

    def read_ingest_insight_pages(
        self, operation_id: str, *, page_count: int, target_count: int, digest: str
    ) -> list[IngestInsightPageHistoricalReceipt]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT detail_json FROM operation_events WHERE operation_id = ? AND event_type = 'ingest_insight_page' "
                "ORDER BY sequence",
                (operation_id,),
            ).fetchall()
        if len(rows) != page_count:
            raise ValueError("ingest insight page count differs from terminal receipt")
        pages = [IngestInsightPageHistoricalReceipt.model_validate_json(str(row[0])) for row in rows]
        if [page.ordinal for page in pages] != list(range(page_count)):
            raise ValueError("ingest insight pages are not contiguous")
        if sum(len(page.targets) for page in pages) != target_count or ingest_insight_pages_digest(pages) != digest:
            raise ValueError("ingest insight pages differ from terminal receipt")
        return pages

    def resolve_ingest_insight_pages(
        self, receipt: IngestHistoricalReceiptV2
    ) -> list[IngestInsightPageHistoricalReceipt]:
        """Return every historical profile target, including referenced pages."""
        if receipt.insight_pages_ref is None:
            return list(receipt.insight_pages)
        assert receipt.insight_pages_digest is not None
        return self.read_ingest_insight_pages(
            receipt.insight_pages_ref,
            page_count=receipt.insight_page_count,
            target_count=receipt.summary.profile_targets_observed,
            digest=receipt.insight_pages_digest,
        )

    @_continuity_mutation("append_ingest_refusal_page")
    def append_ingest_refusal_page(self, operation_id: str, page: IngestRefusalPageHistoricalReceipt) -> None:
        """Retain one page of refused memberships before the terminal receipt cites it."""
        with self._connection() as conn:
            run = conn.execute(
                "SELECT operation_name, status FROM operation_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if run is None or str(run[0]) != INGEST_OPERATION or str(run[1]) == "completed":
                raise ValueError("ingest refusal page lacks an open ingest operation")
            last_row = conn.execute(
                "SELECT detail_json FROM operation_events WHERE operation_id = ? AND event_type = 'ingest_refusal_page' "
                "ORDER BY sequence DESC LIMIT 1",
                (operation_id,),
            ).fetchone()
            last_ordinal = -1 if last_row is None else int(json.loads(str(last_row[0]))["ordinal"])
            payload = page.model_dump(mode="json")
            if page.ordinal <= last_ordinal:
                prior_row = conn.execute(
                    "SELECT detail_json FROM operation_events WHERE operation_id = ? "
                    "AND event_type = 'ingest_refusal_page' AND json_extract(detail_json, '$.ordinal') = ?",
                    (operation_id, page.ordinal),
                ).fetchone()
                prior = None if prior_row is None else json.loads(str(prior_row[0]))
                if prior != payload:
                    raise ValueError("ingest refusal page conflicts with durable page")
                return
            if page.ordinal != last_ordinal + 1:
                raise ValueError("ingest refusal pages must be contiguous")
            self._append_event(
                conn,
                operation_id=operation_id,
                event_type="ingest_refusal_page",
                occurred_at_ms=cast(int, self._command_value("now_ms", int(time.time() * 1000))),
                detail=payload,
            )

    def resolve_ingest_refusals(self, receipt: IngestHistoricalReceiptV2) -> list[IngestRefusedMembershipHistorical]:
        """Return every refused membership a terminal receipt names, inline or paged."""
        summary = receipt.summary
        if summary.refused_membership_pages_ref is None:
            return list(summary.refused_memberships)
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT detail_json FROM operation_events WHERE operation_id = ? AND event_type = 'ingest_refusal_page' "
                "ORDER BY sequence",
                (summary.refused_membership_pages_ref,),
            ).fetchall()
        pages = [IngestRefusalPageHistoricalReceipt.model_validate_json(str(row[0])) for row in rows]
        if (
            [page.ordinal for page in pages] != list(range(summary.refused_membership_page_count))
            or sum(len(page.refusals) for page in pages) != summary.refused_membership_count
            or ingest_refusal_pages_digest(pages) != summary.refused_memberships_digest
        ):
            raise ValueError("ingest refusal pages differ from terminal receipt")
        return [refusal for page in pages for refusal in page.refusals]

    @_continuity_mutation("append_ingest_input_page")
    def append_ingest_input_page(self, operation_id: str, page: IngestInputPageHistoricalReceipt) -> None:
        """Retain one complete input page before an ingest/v2 root cites it."""
        payload = page.model_dump(mode="json")
        with self._connection() as conn:
            run = conn.execute(
                "SELECT operation_name, status FROM operation_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if run is None or str(run[0]) != INGEST_OPERATION or str(run[1]) == "completed":
                raise ValueError("ingest input page lacks an open ingest operation")
            last = conn.execute(
                "SELECT detail_json FROM operation_events WHERE operation_id = ? AND event_type = 'ingest_input_page' "
                "ORDER BY sequence DESC LIMIT 1",
                (operation_id,),
            ).fetchone()
            previous = None if last is None else IngestInputPageHistoricalReceipt.model_validate_json(str(last[0]))
            last_ordinal = -1 if previous is None else previous.ordinal
            if page.ordinal <= last_ordinal:
                prior = conn.execute(
                    "SELECT detail_json FROM operation_events WHERE operation_id = ? AND event_type = 'ingest_input_page' "
                    "AND json_extract(detail_json, '$.ordinal') = ? LIMIT 1",
                    (operation_id, page.ordinal),
                ).fetchone()
                if prior is None or json.loads(str(prior[0])) != payload:
                    raise ValueError("ingest input page conflicts with durable page")
                return
            if page.ordinal != last_ordinal + 1:
                raise ValueError("ingest input pages must be contiguous")
            if previous is not None and previous.items[-1].logical_coordinate >= page.items[0].logical_coordinate:
                raise ValueError("ingest input pages must be globally ordered")
            coordinates = [item.logical_coordinate for item in page.items]
            if coordinates != sorted(set(coordinates)):
                raise ValueError("ingest input page repeats or reorders a coordinate")
            self._append_event(
                conn,
                operation_id=operation_id,
                event_type="ingest_input_page",
                occurred_at_ms=cast(int, self._command_value("now_ms", int(time.time() * 1000))),
                detail=payload,
            )

    @staticmethod
    def _scan_ingest_input_pages(
        conn: sqlite3.Connection, receipt: IngestHistoricalReceiptV2
    ) -> Iterator[IngestInputPageHistoricalReceipt]:
        digest = hashlib.sha256()
        previous_sequence = 0
        previous_coordinate: str | None = None
        item_count = 0
        for ordinal in range(receipt.input_page_count):
            row = conn.execute(
                "SELECT sequence, detail_json FROM operation_events WHERE operation_id = ? "
                "AND event_type = 'ingest_input_page' AND sequence > ? ORDER BY sequence LIMIT 1",
                (receipt.input_pages_ref, previous_sequence),
            ).fetchone()
            if row is None:
                raise ValueError("historical ingest input page is missing")
            previous_sequence = int(row[0])
            page = IngestInputPageHistoricalReceipt.model_validate_json(str(row[1]))
            if page.ordinal != ordinal:
                raise ValueError("historical ingest input pages are reordered")
            coordinates = [item.logical_coordinate for item in page.items]
            if coordinates != sorted(set(coordinates)) or (
                previous_coordinate is not None and previous_coordinate >= coordinates[0]
            ):
                raise ValueError("historical ingest input coordinates repeat or reorder")
            previous_coordinate = coordinates[-1]
            item_count += len(page.items)
            digest.update(f"{page.ordinal}:{page.digest}\n".encode("ascii"))
            yield page
        extra = conn.execute(
            "SELECT 1 FROM operation_events WHERE operation_id = ? AND event_type = 'ingest_input_page' "
            "AND sequence > ? LIMIT 1",
            (receipt.input_pages_ref, previous_sequence),
        ).fetchone()
        if extra is not None or item_count != receipt.input_count or digest.hexdigest() != receipt.input_pages_digest:
            raise ValueError("historical ingest input pages differ from terminal root")

    def iter_ingest_input_pages(self, receipt: IngestHistoricalReceiptV2) -> Iterator[IngestInputPageHistoricalReceipt]:
        """Read the operation-owned denominator one ordered audit page at a time."""
        with self._connection() as conn:
            yield from self._scan_ingest_input_pages(conn, receipt)

    def read_ingest_input_pages(self, receipt: IngestHistoricalReceiptV2) -> list[IngestInputPageHistoricalReceipt]:
        """Resolve the exact historical denominator from audit alone."""
        return list(self.iter_ingest_input_pages(receipt))

    @_continuity_mutation("append_ingest_input_raw_page")
    def append_ingest_input_raw_page(self, operation_id: str, page: IngestInputRawPageHistoricalReceipt) -> None:
        with self._connection() as conn:
            run = conn.execute(
                "SELECT operation_name, status FROM operation_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if run is None or str(run[0]) != INGEST_OPERATION or str(run[1]) == "completed":
                raise ValueError("ingest input raw page lacks an open ingest operation")
            rows = conn.execute(
                "SELECT detail_json FROM operation_events WHERE operation_id = ? "
                "AND event_type = 'ingest_input_raw_page' AND json_extract(detail_json, '$.source_item_id') = ? "
                "ORDER BY sequence DESC LIMIT 1",
                (operation_id, page.source_item_id),
            ).fetchone()
            previous = None if rows is None else json.loads(str(rows[0]))
            last_ordinal = -1 if previous is None else int(previous["ordinal"])
            payload = page.model_dump(mode="json")
            if page.ordinal <= last_ordinal:
                prior_row = conn.execute(
                    "SELECT detail_json FROM operation_events WHERE operation_id = ? "
                    "AND event_type = 'ingest_input_raw_page' "
                    "AND json_extract(detail_json, '$.source_item_id') = ? "
                    "AND json_extract(detail_json, '$.ordinal') = ?",
                    (operation_id, page.source_item_id, page.ordinal),
                ).fetchone()
                prior = None if prior_row is None else json.loads(str(prior_row[0]))
                if prior != payload:
                    raise ValueError("ingest input raw page conflicts with durable page")
                return
            if page.ordinal != last_ordinal + 1:
                raise ValueError("ingest input raw pages must be contiguous")
            if previous is not None and str(previous["raws"][-1]["raw_id"]) >= page.raws[0].raw_id:
                raise ValueError("ingest input raw pages must be globally sorted")
            self._append_event(
                conn,
                operation_id=operation_id,
                event_type="ingest_input_raw_page",
                occurred_at_ms=cast(int, self._command_value("now_ms", int(time.time() * 1000))),
                detail=payload,
            )

    def read_ingest_input_raw_pages(
        self,
        operation_id: str,
        *,
        source_item_id: str,
        page_count: int,
        raw_count: int,
        unresolved_count: int,
        digest: str,
    ) -> list[IngestInputRawPageHistoricalReceipt]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT detail_json FROM operation_events WHERE operation_id = ? "
                "AND event_type = 'ingest_input_raw_page' AND json_extract(detail_json, '$.source_item_id') = ? "
                "ORDER BY sequence",
                (operation_id, source_item_id),
            ).fetchall()
        if len(rows) != page_count:
            raise ValueError("ingest input raw page count differs from terminal receipt")
        pages = [IngestInputRawPageHistoricalReceipt.model_validate_json(str(row[0])) for row in rows]
        ids = [raw.raw_id for page in pages for raw in page.raws]
        if [page.ordinal for page in pages] != list(range(page_count)) or ids != sorted(set(ids)):
            raise ValueError("ingest input raw pages are not contiguous and sorted")
        if (
            len(ids) != raw_count
            or sum(raw.unresolved for page in pages for raw in page.raws) != unresolved_count
            or ingest_input_raw_pages_digest(pages) != digest
        ):
            raise ValueError("ingest input raw pages differ from terminal receipt")
        return pages

    def historical_machine_receipt(self, operation_id: str) -> MachineHistoricalReceipt | None:
        """Return a closed terminal receipt from audit history, never live tiers.

        An absent receipt is meaningful for legacy or interrupted operations.
        A malformed purported receipt is an audit integrity failure, not a
        reason to reconstruct a result from source or index state.
        """

        with self._connection() as conn:
            run = conn.execute("SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone()
            if run is None or str(run[0]) != "completed":
                return None
            event = conn.execute(
                """
                SELECT detail_json FROM operation_events
                WHERE operation_id = ? AND event_type IN ('attempt_finalized', 'recovery_resolved') AND to_state = 'completed'
                ORDER BY sequence DESC LIMIT 1
                """,
                (operation_id,),
            ).fetchone()
        if event is None:
            return None
        try:
            detail = json.loads(str(event[0]))
        except json.JSONDecodeError as exc:
            raise ValueError("terminal audit event detail is malformed") from exc
        if not isinstance(detail, dict):
            raise ValueError("terminal audit event detail is not an object")
        raw = detail.get("historical_receipt")
        if raw is None:
            return None
        return decode_machine_receipt(raw)

    @staticmethod
    def _append_event(
        conn: sqlite3.Connection,
        *,
        operation_id: str,
        event_type: str,
        occurred_at_ms: int,
        detail: Mapping[str, object] | CanonicalAuditLiteral,
        target_ordinal: int | None = None,
        attempt_id: str | None = None,
        from_state: str | None = None,
        to_state: str | None = None,
        actor_ref: str | None = None,
    ) -> None:
        from polylogue.storage.io_phase_metrics import connection_cursor

        with connection_cursor(
            conn, "SELECT COALESCE(MAX(sequence), 0) + 1 FROM operation_events WHERE operation_id = ?", (operation_id,)
        ) as cursor:
            sequence = int(cursor.fetchone()[0])
        with connection_cursor(
            conn,
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
                "{}"
                if isinstance(detail, CanonicalAuditLiteral)
                else json.dumps(dict(detail), sort_keys=True, separators=(",", ":")),
            ),
        ):
            pass
        if isinstance(detail, CanonicalAuditLiteral):
            from polylogue.storage.sqlite.audit_continuity import write_canonical_audit_literal

            with connection_cursor(
                conn, "SELECT rowid FROM operation_events WHERE operation_id=? AND sequence=?", (operation_id, sequence)
            ) as cursor:
                rowid = int(cursor.fetchone()[0])
            write_canonical_audit_literal(conn, "operation_events", "detail_json", rowid, detail)

    def iter_machine_parts(self, binding: MachineRequestBinding) -> Iterator[dict[str, object]]:
        """Read durable parts in bounded pages without holding a reader over writes."""
        if self.machine_request(binding) is None:
            return
        after = -1
        while True:
            with self._connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM machine_request_parts WHERE archive_identity = ? AND request_id = ? "
                    "AND ordinal > ? ORDER BY ordinal LIMIT ?",
                    (binding.archive_identity, binding.request_id, after, MACHINE_PAGE_PARTS),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                after = int(row["ordinal"])
                yield dict(row)

    def machine_preview_origin(self, binding: MachineRequestBinding) -> str | None:
        """Recover the exact source preview phase from durable shared preview refs."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT origin.request_id FROM machine_request_parts current "
                "JOIN machine_request_parts source ON source.preview_ref = current.preview_ref "
                "JOIN machine_requests origin ON origin.request_id = source.request_id AND origin.archive_identity = source.archive_identity "
                "WHERE current.archive_identity = ? AND current.request_id = ? "
                "AND origin.operation_name = ? AND origin.artifact_kind = 'preview-batch' LIMIT 2",
                (
                    binding.archive_identity,
                    binding.request_id,
                    "mutation.identity-reset.preview"
                    if binding.operation_name.startswith("mutation.identity-reset")
                    else "mutation.session.delete.preview",
                ),
            ).fetchall()
        if len(rows) != 1:
            return None
        return str(rows[0][0])

    def machine_user_change_summary(self, binding: MachineRequestBinding) -> dict[str, object]:
        """Count selected sessions and changed families from durable plan/run facts."""
        population = """WITH selected(session_id) AS (
            SELECT substr(t.target_ref, 9) FROM machine_request_parts p
            JOIN operation_preview_targets t ON t.preview_id = p.preview_ref
            WHERE p.archive_identity = ? AND p.request_id = ? AND t.target_ref LIKE 'session:%'
            UNION
            SELECT json_extract(v.plan_json, '$.context.owner_session_id') FROM machine_request_parts p
            JOIN operation_previews v ON v.preview_id = p.preview_ref
            WHERE p.archive_identity = ? AND p.request_id = ?
              AND json_extract(v.plan_json, '$.context.owner_session_id') IS NOT NULL
        ) """
        args = (binding.archive_identity, binding.request_id) * 2
        with self._connection() as conn:
            count = int(conn.execute(population + "SELECT COUNT(*) FROM selected", args).fetchone()[0])
            sample = [
                str(row[0])
                for row in conn.execute(
                    population + "SELECT session_id FROM selected ORDER BY session_id LIMIT 5", args
                )
            ]
            changed = dict(
                conn.execute(
                    "SELECT r.operation_name, SUM(r.affected_count) FROM machine_request_parts p "
                    "JOIN operation_runs r ON r.operation_id = p.operation_id "
                    "WHERE p.archive_identity = ? AND p.request_id = ? GROUP BY r.operation_name",
                    (binding.archive_identity, binding.request_id),
                )
            )
        return {
            "session_count": count,
            "session_ids_sample": sample,
            "tag_count": int(changed.get("mutate-bulk-tag-sessions", 0)),
            "applied_count": int(changed.get("mutate-bulk-set-metadata", 0)),
        }


__all__ = ["AuditRepository", "AuditTargetState", "CanonicalAuditLiteral", "plan_from_stored_payload", "token_sha256"]

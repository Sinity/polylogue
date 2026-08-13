"""MutationTransaction: the executable PREPARE -> AUTHORIZE -> EXECUTE lifecycle.

Implements the reconciled architecture for polylogue-t46.9 (make
``OperationSpec`` the executable mutation authority) and polylogue-kwsb.2
(``MutationTransaction``: authorize and receipt every destructive operation).

Architecture decision (recorded here, not only in the PR body, so the design
survives independent of any one PR narrative):

``OperationSpec`` (``operations/specs.py``) and ``CliActionContract``
(``operations/action_contracts.py``) are *declarations* -- they say what an
operation is (capability shape, destructive class, surfaces, effects). They do
not execute anything themselves. ``MutationTransaction`` is the *runtime
lifecycle* that turns a declared destructive/mutating operation into a proven
execution: it is ``OperationExecutor``'s protocol for destructive-class
operations specifically. There is exactly one executable authority:

* A domain module owns an :class:`OperationSpec`-equivalent to describe
  intent, and a :class:`MutationActuator` implementation to describe
  mechanism (``prepare`` = resolve targets from live state with zero
  mutation; ``apply`` = perform the real mutation and return a receipt).
* :class:`OperationExecutor` is the single place that runs PREPARE, checks
  a caller-declared confirmation strength against the actuator's declared
  floor, binds an :class:`MutationAuthorization` to the prepared plan's
  hash, and revalidates that hash against a *fresh* PREPARE immediately
  before EXECUTE actually mutates anything. No adapter (CLI/MCP/API/daemon)
  may call ``actuator.apply`` directly -- only ``OperationExecutor.execute``
  may, and it always re-resolves and re-hashes the plan first.

This module intentionally does not become a second dispatch table or a
generic "run any handler" facility. ``ArchiveWriteGateway``
(``archive/write_effects.py``, owned by polylogue-a7xr.18) remains a distinct,
narrower ingest-commit effects gateway; storage-layer excision guards remain
defense in depth. ``MutationTransaction`` is strictly the authorization/
preview/receipt layer that every destructive surface must pass through before
reaching those lower layers.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from polylogue.operations.audit import AuditRepository
    from polylogue.operations.bindings import OperationBinding

#: Destructive/mutating classification. Ordered roughly by blast radius:
#: ``reversible`` writes (tags/metadata) can be undone by another write;
#: ``reset`` tombstones rebuildable rows while preserving durable evidence;
#: ``delete`` permanently removes archive rows but re-ingest of the original
#: source can resurrect them; ``excise`` is the durable, cross-tier,
#: re-ingest-proof removal (right-to-forget); ``maintenance`` rewrites
#: rebuildable derived state without removing authored evidence.
DestructiveClass = Literal["additive", "reversible", "maintenance", "reset", "delete", "excise"]

Surface = Literal["cli", "api", "mcp", "daemon", "maintenance", "internal"]
IdempotencyPolicy = Literal["none", "effect_key", "convergent", "compare_and_set"]
RecoveryPolicy = Literal[
    "rebuild",
    "restore_verified_backup",
    "reauthenticate",
    "retry_convergent",
    "reconcile_required",
    "none",
]
TargetDurability = Literal["durable", "derived", "disposable", "external"]

#: Confirmation strength a caller can present at AUTHORIZE time, ordered
#: weakest to strongest. Each :class:`MutationActuator` declares the floor it
#: requires; ``OperationExecutor.authorize`` refuses anything weaker.
#:
#: - ``role_only``: the caller's role/capability alone (reversible writes).
#: - ``confirm_flag``: an explicit boolean/CLI ``--yes`` (interim jn40-style
#:   mitigation, still accepted for delete/reset/excise while a fuller
#:   client-held preview-token flow is Phase 2 debt -- see the PR body).
#: - ``bound_token``: a caller-supplied plan hash that must match a fresh
#:   PREPARE, i.e. proof the caller actually observed *this* plan
#:   (dry-run/preview output) before authorizing it.
ConfirmationStrength = Literal["role_only", "confirm_flag", "bound_token"]

_STRENGTH_ORDER: dict[ConfirmationStrength, int] = {
    "role_only": 0,
    "confirm_flag": 1,
    "bound_token": 2,
}

_CLASS_ORDER: dict[DestructiveClass, int] = {
    "additive": 0,
    "reversible": 1,
    "maintenance": 2,
    "reset": 3,
    "delete": 4,
    "excise": 5,
}

#: Per-target outcome vocabulary for a mutation receipt (kwsb.2 AC3).
#: ``unknown`` is reserved for crash/timeout paths where the actuator cannot
#: prove the mutation did or did not apply; it must never be silently
#: upgraded to ``applied``.
MutationTargetStatus = Literal["applied", "already_satisfied", "blocked", "failed", "unknown"]


class MutationTransactionError(RuntimeError):
    """Base class for MutationTransaction protocol violations."""


class ConfirmationRequiredError(MutationTransactionError):
    """AUTHORIZE was attempted with a confirmation strength below the actuator's floor."""


class PlanStaleError(MutationTransactionError):
    """EXECUTE's fresh PREPARE no longer matches the authorized plan hash.

    This is the structural refusal that makes a stale or tampered
    authorization unusable: the live target set moved between AUTHORIZE and
    EXECUTE (TOCTOU), so the bound plan hash can no longer be trusted to
    describe what would actually be mutated.
    """


class AuthorizationMismatchError(MutationTransactionError):
    """EXECUTE was attempted with an authorization bound to a different plan."""


class CapabilityDeniedError(MutationTransactionError):
    """The authenticated principal lacks an exact capability declared by the spec."""


class SurfaceDeniedError(MutationTransactionError):
    """The operation is not declared for the requesting surface."""


class TokenExpiredError(MutationTransactionError):
    """A bound authorization is outside its short validity window."""


class TokenConsumedError(MutationTransactionError):
    """A bound authorization was already consumed."""


class AuditFinalizationError(MutationTransactionError):
    """The domain result could not be durably finalized in audit.db."""


@dataclass(frozen=True, slots=True)
class MutationPrincipal:
    """Authenticated identity and exact capabilities for one mutation request."""

    actor_ref: str
    capabilities: frozenset[str]
    surface: Surface
    role_label: str | None = None

    def __post_init__(self) -> None:
        if not self.actor_ref:
            raise ValueError("mutation principal actor_ref must not be empty")
        if any(not capability for capability in self.capabilities):
            raise ValueError("mutation principal capabilities must not contain empty values")


@dataclass(frozen=True, slots=True)
class TargetAuthorityPolicy:
    """Spec-owned policy for one closed target-key vocabulary entry."""

    key: str
    target_kinds: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    destructive_class: DestructiveClass
    required_confirmation: ConfirmationStrength
    allowed_durabilities: tuple[TargetDurability, ...]
    allowed_recovery: tuple[RecoveryPolicy, ...]

    def __post_init__(self) -> None:
        if not self.key or not self.target_kinds or not self.required_capabilities:
            raise ValueError("target authority policies require key, target kinds, and capabilities")
        if len(set(self.required_capabilities)) != len(self.required_capabilities):
            raise ValueError(f"duplicate capabilities in target policy {self.key!r}")


@dataclass(frozen=True, slots=True)
class MutationTarget:
    """Canonical typed target included in a prepared plan and audit rows."""

    kind: str
    ref: str
    policy_key: str
    identity_digest: str
    effect_identity: str
    durability: TargetDurability
    recovery: RecoveryPolicy

    def canonical_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "ref": self.ref,
            "policy_key": self.policy_key,
            "identity_digest": self.identity_digest,
            "effect_identity": self.effect_identity,
            "durability": self.durability,
            "recovery": self.recovery,
        }


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_document(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def compute_target_digest(targets: tuple[MutationTarget, ...]) -> str:
    """Hash ordered typed targets without stringifying arbitrary objects."""

    return _sha256_document([target.canonical_dict() for target in targets])


def compute_parameter_digest(raw_plan: MutationPlan) -> str:
    """Hash stable caller intent without the clock-bound preview envelope."""

    return _sha256_document(
        {
            "operation": raw_plan.operation,
            "destructive_class": raw_plan.destructive_class,
            "affected_tiers": list(raw_plan.affected_tiers),
            "context": {key: raw_plan.context[key] for key in sorted(raw_plan.context)},
        }
    )


def compute_typed_plan_hash(
    *,
    operation: str,
    operation_version: int,
    archive_instance_id: str,
    archive_identity_digest: str,
    parameter_digest: str,
    target_digest: str,
    required_capabilities: tuple[str, ...],
    destructive_class: DestructiveClass,
    required_confirmation: ConfirmationStrength,
    affected_tiers: tuple[str, ...],
    context: Mapping[str, object],
) -> str:
    """Hash every authority-relevant field of a typed mutation plan."""

    return _sha256_document(
        {
            "operation": operation,
            "operation_version": operation_version,
            "archive_instance_id": archive_instance_id,
            "archive_identity_digest": archive_identity_digest,
            "parameter_digest": parameter_digest,
            "target_digest": target_digest,
            "required_capabilities": list(required_capabilities),
            "destructive_class": destructive_class,
            "required_confirmation": required_confirmation,
            "affected_tiers": list(affected_tiers),
            "context": {key: context[key] for key in sorted(context)},
        }
    )


def compute_plan_hash(
    *,
    operation: str,
    target_refs: tuple[str, ...],
    affected_tiers: tuple[str, ...],
    destructive_class: DestructiveClass,
    context: Mapping[str, object],
) -> str:
    """Return a stable content hash binding an operation to its exact plan.

    The hash covers everything that changing would mean "a different plan":
    the operation identity, the exact resolved target set, the tiers it
    would touch, its destructive class, and any operation-specific context
    (e.g. ``cascade_lineage``). It deliberately excludes timestamps/actor
    identity -- those belong to the authorization, not the plan.
    """

    payload = {
        "operation": operation,
        "target_refs": sorted(target_refs),
        "affected_tiers": sorted(affected_tiers),
        "destructive_class": destructive_class,
        "context": {key: context[key] for key in sorted(context)},
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MutationPlan:
    """PREPARE output: a bounded, hashable, zero-mutation preview.

    ``target_refs`` are exact resolved object refs (e.g. ``session:<id>``),
    never raw caller tokens -- resolution (prefix matching, typo handling,
    source-path lookup) happens before the plan is built so the plan hash
    binds to reality, not to the caller's possibly-ambiguous input.
    """

    operation: str
    destructive_class: DestructiveClass
    target_refs: tuple[str, ...]
    affected_tiers: tuple[str, ...]
    reversible: bool
    prepared_at: str
    plan_hash: str
    context: Mapping[str, object] = field(default_factory=dict)
    operation_version: int = 1
    archive_instance_id: str = "legacy"
    archive_identity_digest: str = "legacy"
    required_capabilities: tuple[str, ...] = ()
    required_confirmation: ConfirmationStrength = "role_only"
    targets: tuple[MutationTarget, ...] = ()
    parameter_digest: str = ""
    target_digest: str = ""
    prepared_at_ms: int = 0
    expires_at_ms: int = 0

    @property
    def target_count(self) -> int:
        return len(self.target_refs)

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "destructive_class": self.destructive_class,
            "target_refs": list(self.target_refs),
            "affected_tiers": list(self.affected_tiers),
            "reversible": self.reversible,
            "prepared_at": self.prepared_at,
            "plan_hash": self.plan_hash,
            "target_count": self.target_count,
            "context": dict(self.context),
            "operation_version": self.operation_version,
            "archive_instance_id": self.archive_instance_id,
            "archive_identity_digest": self.archive_identity_digest,
            "required_capabilities": list(self.required_capabilities),
            "required_confirmation": self.required_confirmation,
            "targets": [target.canonical_dict() for target in self.targets],
            "parameter_digest": self.parameter_digest,
            "target_digest": self.target_digest,
            "prepared_at_ms": self.prepared_at_ms,
            "expires_at_ms": self.expires_at_ms,
        }


@dataclass(frozen=True, slots=True)
class MutationPreview:
    """Durable preview reference and its immutable typed plan."""

    preview_ref: str
    plan: MutationPlan


def build_typed_plan(
    *,
    operation: str,
    operation_version: int,
    archive_instance_id: str,
    archive_identity_digest: str,
    targets: tuple[MutationTarget, ...],
    affected_tiers: tuple[str, ...],
    parameter_digest: str,
    required_capabilities: tuple[str, ...],
    destructive_class: DestructiveClass,
    required_confirmation: ConfirmationStrength,
    prepared_at_ms: int,
    expires_at_ms: int,
    context: Mapping[str, object] | None = None,
) -> MutationPlan:
    """Construct a plan whose hash covers the complete typed authority input."""

    target_digest = compute_target_digest(targets)
    plan_hash = compute_typed_plan_hash(
        operation=operation,
        operation_version=operation_version,
        archive_instance_id=archive_instance_id,
        archive_identity_digest=archive_identity_digest,
        parameter_digest=parameter_digest,
        target_digest=target_digest,
        required_capabilities=required_capabilities,
        destructive_class=destructive_class,
        required_confirmation=required_confirmation,
        affected_tiers=affected_tiers,
        context=context or {},
    )
    return MutationPlan(
        operation=operation,
        destructive_class=destructive_class,
        target_refs=tuple(target.ref for target in targets),
        affected_tiers=affected_tiers,
        reversible=destructive_class in {"additive", "reversible"},
        prepared_at=datetime.fromtimestamp(prepared_at_ms / 1000, UTC).isoformat(),
        plan_hash=plan_hash,
        context=dict(context or {}),
        operation_version=operation_version,
        archive_instance_id=archive_instance_id,
        archive_identity_digest=archive_identity_digest,
        required_capabilities=required_capabilities,
        required_confirmation=required_confirmation,
        targets=targets,
        parameter_digest=parameter_digest,
        target_digest=target_digest,
        prepared_at_ms=prepared_at_ms,
        expires_at_ms=expires_at_ms,
    )


def validate_mutation_plan_integrity(plan: MutationPlan) -> None:
    """Reject a reconstructed preview whose typed authority fields were changed."""

    target_refs = tuple(target.ref for target in plan.targets)
    target_digest = compute_target_digest(plan.targets)
    plan_hash = compute_typed_plan_hash(
        operation=plan.operation,
        operation_version=plan.operation_version,
        archive_instance_id=plan.archive_instance_id,
        archive_identity_digest=plan.archive_identity_digest,
        parameter_digest=plan.parameter_digest,
        target_digest=target_digest,
        required_capabilities=plan.required_capabilities,
        destructive_class=plan.destructive_class,
        required_confirmation=plan.required_confirmation,
        affected_tiers=plan.affected_tiers,
        context=plan.context,
    )
    if plan.target_refs != target_refs or plan.target_digest != target_digest or plan.plan_hash != plan_hash:
        raise AuthorizationMismatchError("preview plan payload does not match its authority hash")


def build_plan(
    *,
    operation: str,
    destructive_class: DestructiveClass,
    target_refs: tuple[str, ...],
    affected_tiers: tuple[str, ...],
    reversible: bool,
    context: Mapping[str, object] | None = None,
) -> MutationPlan:
    """Construct a :class:`MutationPlan` with a freshly computed plan hash."""

    resolved_context = dict(context or {})
    plan_hash = compute_plan_hash(
        operation=operation,
        target_refs=target_refs,
        affected_tiers=affected_tiers,
        destructive_class=destructive_class,
        context=resolved_context,
    )
    return MutationPlan(
        operation=operation,
        destructive_class=destructive_class,
        target_refs=target_refs,
        affected_tiers=affected_tiers,
        reversible=reversible,
        prepared_at=_utcnow_iso(),
        plan_hash=plan_hash,
        context=resolved_context,
    )


@dataclass(frozen=True, slots=True)
class MutationAuthorization:
    """AUTHORIZE output: actor/role/capability bound to one exact plan hash."""

    plan_hash: str
    actor: str
    role: str
    capability: str
    confirmation_strength: ConfirmationStrength
    authorized_at: str
    preview_ref: str | None = None
    authorization_id: str | None = None
    token: str | None = None
    expires_at_ms: int | None = None
    capabilities: tuple[str, ...] = ()
    surface: Surface | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_hash": self.plan_hash,
            "actor": self.actor,
            "role": self.role,
            "capability": self.capability,
            "confirmation_strength": self.confirmation_strength,
            "authorized_at": self.authorized_at,
            "preview_ref": self.preview_ref,
            "authorization_id": self.authorization_id,
            "expires_at_ms": self.expires_at_ms,
            "capabilities": list(self.capabilities),
            "surface": self.surface,
        }


@dataclass(frozen=True, slots=True)
class MutationReceipt:
    """EXECUTE output: a typed, auditable record of what actually happened."""

    operation: str
    plan_hash: str
    status: MutationTargetStatus
    target_refs: tuple[str, ...]
    affected_count: int
    detail: str | None
    receipt_ref: str | None
    applied_at: str
    domain_receipt: Mapping[str, object] = field(default_factory=dict)
    operation_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "plan_hash": self.plan_hash,
            "status": self.status,
            "target_refs": list(self.target_refs),
            "affected_count": self.affected_count,
            "detail": self.detail,
            "receipt_ref": self.receipt_ref,
            "applied_at": self.applied_at,
            "domain_receipt": dict(self.domain_receipt),
            "operation_id": self.operation_id,
        }


#: Both ``prepare`` and ``apply`` take the *same* argument shape in every
#: actuator this module ships (the domain re-resolves from the same inputs
#: at both PREPARE and EXECUTE-revalidation time) -- one contravariant
#: TypeVar rather than two keeps the protocol's variance sound under mypy
#: --strict (args only ever appear in parameter/input position).
ArgsT = TypeVar("ArgsT", contravariant=True)


@runtime_checkable
class MutationActuator(Protocol[ArgsT]):
    """Domain-owned target resolution (PREPARE) and real mutation (APPLY).

    An actuator never enforces authorization itself -- that is
    :class:`OperationExecutor`'s job. An actuator's ``prepare`` must be safe
    to call any number of times (including immediately before ``apply``, to
    revalidate) and must never mutate state.

    Declared as read-only ``@property`` members (rather than plain mutable
    attributes) so frozen-dataclass actuator implementations -- the intended
    shape, since an actuator's declared identity/policy must not be mutable
    at runtime -- satisfy the protocol under ``mypy --strict``.
    """

    @property
    def operation(self) -> str: ...

    @property
    def destructive_class(self) -> DestructiveClass: ...

    @property
    def required_confirmation(self) -> ConfirmationStrength: ...

    def prepare(self, args: ArgsT) -> MutationPlan: ...

    def apply(self, plan: MutationPlan, args: ArgsT) -> MutationReceipt: ...


class OperationExecutor:
    """The single executable mutation authority: PREPARE -> AUTHORIZE -> EXECUTE.

    Every destructive/mutating surface route (CLI, MCP, API, daemon,
    maintenance) that has been migrated to this protocol constructs one
    :class:`MutationActuator` for its domain and drives it exclusively
    through this class. No adapter calls ``actuator.apply`` directly.
    """

    def __init__(
        self,
        *,
        audit: AuditRepository | None = None,
        now_ms: Callable[[], int] | None = None,
        token_factory: Callable[[], str] | None = None,
        archive_root: Path | None = None,
    ) -> None:
        self._audit = audit
        self._now_ms = now_ms or (lambda: int(datetime.now(UTC).timestamp() * 1000))
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(32))
        self._archive_root = archive_root

    @classmethod
    def for_archive_root(
        cls,
        archive_root: Path,
        *,
        now_ms: Callable[[], int] | None = None,
        token_factory: Callable[[], str] | None = None,
    ) -> OperationExecutor:
        """Compose production mutation execution with the archive's audit tier."""

        from polylogue.operations.audit import AuditRepository

        audit = AuditRepository.for_archive_root(
            archive_root,
            attempt_owner_id=AuditRepository.current_process_attempt_owner(),
        )
        audit.reconcile_continuity()
        audit.recover_abandoned_attempts()
        return cls(audit=audit, now_ms=now_ms, token_factory=token_factory, archive_root=archive_root)

    def prepare(self, actuator: MutationActuator[ArgsT], args: ArgsT) -> MutationPlan:
        """PREPARE: resolve exact targets from live state. Never mutates."""

        return actuator.prepare(args)

    def prepare_bound(
        self,
        binding: OperationBinding[ArgsT, object],
        args: ArgsT,
        principal: MutationPrincipal,
        *,
        archive_instance_id: str,
        archive_identity_digest: str,
        parameter_digest: str,
        expires_at_ms: int | None = None,
        raw_plan: MutationPlan | None = None,
    ) -> MutationPreview:
        """Prepare and durably record a versioned, capability-bound preview."""

        binding.validate()
        if principal.surface not in binding.spec.allowed_surfaces:
            raise SurfaceDeniedError(f"{binding.spec.name!r} is not allowed on {principal.surface!r}")
        plan = self._typed_plan_from_actuator(
            binding,
            raw_plan or binding.actuator.prepare(args),
            archive_instance_id=archive_instance_id,
            archive_identity_digest=archive_identity_digest,
            parameter_digest=parameter_digest,
            expires_at_ms=expires_at_ms or self._now_ms() + 60_000,
        )
        if self._audit is None:
            return MutationPreview(preview_ref=f"preview:{plan.plan_hash}", plan=plan)
        preview_ref = self._audit.create_preview(plan, principal)
        return MutationPreview(preview_ref=preview_ref, plan=plan)

    def prepare_bound_for_archive(
        self,
        binding: OperationBinding[ArgsT, object],
        args: ArgsT,
        principal: MutationPrincipal,
        *,
        archive_root: Path,
    ) -> MutationPreview:
        """Prepare a production mutation with live archive and audit authority."""

        if self._audit is None:
            raise MutationTransactionError("production mutation preparation requires a durable audit repository")
        from polylogue.storage.archive_identity import ArchiveIdentity

        raw_plan = binding.actuator.prepare(args)
        return self.prepare_bound(
            binding,
            args,
            principal,
            archive_instance_id=self._audit.ensure_archive_authority(now_ms=self._now_ms()),
            archive_identity_digest=ArchiveIdentity.resolve(archive_root).authority_identity_digest,
            parameter_digest=compute_parameter_digest(raw_plan),
            raw_plan=raw_plan,
        )

    def authorize_bound(
        self,
        binding: OperationBinding[ArgsT, object],
        preview: MutationPreview,
        principal: MutationPrincipal,
        *,
        confirmation_strength: ConfirmationStrength | None = None,
    ) -> MutationAuthorization:
        """Issue a random one-time token bound to the persisted preview."""

        binding.validate()
        validate_mutation_plan_integrity(preview.plan)
        plan = preview.plan
        required = set(plan.required_capabilities)
        if not required.issubset(principal.capabilities):
            missing = sorted(required - principal.capabilities)
            raise CapabilityDeniedError(f"principal lacks declared capabilities: {missing}")
        if self._now_ms() >= plan.expires_at_ms:
            raise TokenExpiredError("cannot authorize an expired preview")
        strength = confirmation_strength or plan.required_confirmation
        if _STRENGTH_ORDER[strength] < _STRENGTH_ORDER[plan.required_confirmation]:
            raise ConfirmationRequiredError(
                f"{binding.spec.name!r} requires {plan.required_confirmation!r}, got {strength!r}"
            )
        token = self._token_factory()
        authorization = MutationAuthorization(
            plan_hash=plan.plan_hash,
            actor=principal.actor_ref,
            role=principal.role_label or "",
            capability=plan.required_capabilities[0] if plan.required_capabilities else "",
            confirmation_strength=strength,
            authorized_at=_utcnow_iso(),
            preview_ref=preview.preview_ref,
            token=token,
            expires_at_ms=plan.expires_at_ms,
            capabilities=tuple(sorted(required)),
            surface=principal.surface,
        )
        if self._audit is not None:
            authorization_id = self._audit.issue_authorization(
                preview, principal, authorization, issued_at_ms=self._now_ms()
            )
            authorization = replace(authorization, authorization_id=authorization_id)
        return authorization

    def execute_bound(
        self,
        binding: OperationBinding[ArgsT, object],
        preview: MutationPreview,
        authorization: MutationAuthorization,
        args: ArgsT,
    ) -> MutationReceipt:
        """Consume a bound token, journal intent, apply, and finalize honestly."""

        binding.validate()
        validate_mutation_plan_integrity(preview.plan)
        if authorization.preview_ref != preview.preview_ref or authorization.token is None:
            raise AuthorizationMismatchError("authorization is not bound to this preview")
        if (
            self._audit is None
            and authorization.expires_at_ms is not None
            and self._now_ms() >= authorization.expires_at_ms
        ):
            raise TokenExpiredError("authorization token is expired")
        if self._archive_root is not None:
            from polylogue.storage.archive_identity import ArchiveIdentity

            live_identity = ArchiveIdentity.resolve(self._archive_root).authority_identity_digest
            if live_identity != preview.plan.archive_identity_digest:
                raise PlanStaleError("archive identity changed after the bound preview was prepared")
        fresh_plan = self._typed_plan_from_actuator(
            binding,
            binding.actuator.prepare(args),
            archive_instance_id=preview.plan.archive_instance_id,
            archive_identity_digest=preview.plan.archive_identity_digest,
            parameter_digest=preview.plan.parameter_digest,
            expires_at_ms=preview.plan.expires_at_ms,
        )
        if fresh_plan.plan_hash != preview.plan.plan_hash:
            if self._audit is not None:
                self._audit.mark_preview_stale(preview)
            raise PlanStaleError(
                f"{binding.spec.name!r} preview {preview.plan.plan_hash!r} is stale; "
                f"live state now resolves to {fresh_plan.plan_hash!r}"
            )
        operation_id: str | None = None
        if self._audit is not None:
            operation_id = self._audit.consume_authorization_and_start(preview, authorization)
        try:
            result = binding.actuator.apply(fresh_plan, args)
        except Exception as exc:
            if self._audit is not None and operation_id is not None:
                self._audit.finalize_attempt(
                    operation_id,
                    status="unknown",
                    error_summary=str(exc)[:512],
                    unknown_reason="actuator exception after durable intent",
                )
            raise
        receipt = result
        if self._audit is not None and operation_id is not None:
            try:
                self._audit.finalize_attempt(operation_id, status=receipt.status, receipt=receipt)
            except Exception as exc:
                raise AuditFinalizationError(
                    "domain effect is not reported completed without audit finalization"
                ) from exc
            receipt = replace(
                receipt,
                receipt_ref=f"mutation-operation:{operation_id}",
                operation_id=operation_id,
            )
        return receipt

    def reconcile_operation(
        self,
        operation_id: str,
        *,
        outcome: Literal["applied", "absent", "unknown"],
        domain_receipt_ref: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Record explicit reconciliation without inferring success from a crash."""

        if self._audit is None:
            raise MutationTransactionError("reconciliation requires a durable audit repository")
        self._audit.reconcile_attempt(
            operation_id,
            outcome=outcome,
            domain_receipt_ref=domain_receipt_ref,
            reason=reason,
        )

    def find_interrupted_operation(self, *, operation_name: str, parameter_digest: str) -> str | None:
        """Find the uniquely identified interrupted durable attempt for a recovery route."""

        if self._audit is None:
            raise MutationTransactionError("interrupted-operation lookup requires a durable audit repository")
        return self._audit.find_interrupted_operation(
            operation_name=operation_name,
            parameter_digest=parameter_digest,
        )

    def _typed_plan_from_actuator(
        self,
        binding: OperationBinding[ArgsT, object],
        plan: MutationPlan,
        *,
        archive_instance_id: str,
        archive_identity_digest: str,
        parameter_digest: str,
        expires_at_ms: int,
    ) -> MutationPlan:
        policies = {policy.key: policy for policy in binding.spec.target_authority}
        targets = plan.targets
        if not targets:
            default = next(iter(policies.values()), None)
            if default is None:
                raise MutationTransactionError(f"{binding.spec.name!r} has no target authority policy")
            if len(default.allowed_durabilities) != 1 or len(default.allowed_recovery) != 1:
                raise MutationTransactionError(
                    f"{binding.spec.name!r} must emit typed targets for an ambiguous default authority policy"
                )
            targets = tuple(
                MutationTarget(
                    kind=ref.split(":", 1)[0],
                    ref=ref,
                    policy_key=default.key,
                    identity_digest=_sha256_document({"ref": ref}),
                    effect_identity=f"{binding.spec.name}:{ref}",
                    durability=default.allowed_durabilities[0],
                    recovery=default.allowed_recovery[0],
                )
                for ref in plan.target_refs
            )
        for target in targets:
            policy = policies.get(target.policy_key)
            if policy is None:
                raise MutationTransactionError(f"actuator returned unregistered target policy {target.policy_key!r}")
            if target.kind not in policy.target_kinds:
                raise MutationTransactionError(f"target kind {target.kind!r} is not allowed by {policy.key!r}")
            if target.durability not in policy.allowed_durabilities:
                raise MutationTransactionError(
                    f"target durability {target.durability!r} is not allowed by {policy.key!r}"
                )
            if target.recovery not in policy.allowed_recovery:
                raise MutationTransactionError(f"target recovery {target.recovery!r} is not allowed by {policy.key!r}")
        required_capabilities = tuple(
            sorted(
                {capability for target in targets for capability in policies[target.policy_key].required_capabilities}
            )
        )
        destructive_class = max(
            (policies[target.policy_key].destructive_class for target in targets),
            key=lambda value: _CLASS_ORDER[value],
            default=plan.destructive_class,
        )
        required_confirmation = max(
            (policies[target.policy_key].required_confirmation for target in targets),
            key=lambda value: _STRENGTH_ORDER[value],
            default=plan.required_confirmation,
        )
        return build_typed_plan(
            operation=binding.spec.name,
            operation_version=binding.spec.operation_version,
            archive_instance_id=archive_instance_id,
            archive_identity_digest=archive_identity_digest,
            targets=targets,
            affected_tiers=binding.spec.affected_tiers or plan.affected_tiers,
            parameter_digest=parameter_digest,
            required_capabilities=required_capabilities,
            destructive_class=destructive_class,
            required_confirmation=required_confirmation,
            prepared_at_ms=self._now_ms(),
            expires_at_ms=expires_at_ms,
            context=plan.context,
        )

    def authorize(
        self,
        actuator: MutationActuator[ArgsT],
        plan: MutationPlan,
        *,
        actor: str,
        role: str,
        capability: str,
        confirmation_strength: ConfirmationStrength,
    ) -> MutationAuthorization:
        """AUTHORIZE: bind actor/role/capability + confirmation to the plan hash.

        Refuses (:class:`ConfirmationRequiredError`) when the presented
        confirmation strength is weaker than the actuator's declared floor
        for its destructive class.
        """

        if _STRENGTH_ORDER[confirmation_strength] < _STRENGTH_ORDER[actuator.required_confirmation]:
            raise ConfirmationRequiredError(
                f"{actuator.operation!r} requires confirmation strength "
                f"{actuator.required_confirmation!r}, got {confirmation_strength!r}"
            )
        return MutationAuthorization(
            plan_hash=plan.plan_hash,
            actor=actor,
            role=role,
            capability=capability,
            confirmation_strength=confirmation_strength,
            authorized_at=_utcnow_iso(),
        )

    def execute(
        self,
        actuator: MutationActuator[ArgsT],
        plan: MutationPlan,
        authorization: MutationAuthorization,
        args: ArgsT,
    ) -> MutationReceipt:
        """EXECUTE: revalidate the plan against live state, then apply.

        Raises :class:`AuthorizationMismatchError` if ``authorization`` was
        bound to a different plan hash than ``plan``, and
        :class:`PlanStaleError` if a fresh PREPARE no longer matches --
        i.e. the live target set moved between AUTHORIZE and EXECUTE.
        """

        if authorization.plan_hash != plan.plan_hash:
            raise AuthorizationMismatchError(
                f"authorization bound to plan {authorization.plan_hash!r} does not match plan {plan.plan_hash!r}"
            )
        fresh_plan = actuator.prepare(args)
        if fresh_plan.plan_hash != plan.plan_hash:
            raise PlanStaleError(
                f"{actuator.operation!r} plan {plan.plan_hash!r} is stale; "
                f"live state now resolves to {fresh_plan.plan_hash!r} "
                f"({fresh_plan.target_count} target(s) vs {plan.target_count})"
            )
        return actuator.apply(plan, args)


def make_target_ref(kind: Literal["session", "message", "block", "source", "index"], value: object) -> str:
    """Return a stable ``kind:value`` target ref, the shared vocabulary for plans/receipts."""

    return f"{kind}:{value}"


__all__ = [
    "AuthorizationMismatchError",
    "AuditFinalizationError",
    "CapabilityDeniedError",
    "ConfirmationRequiredError",
    "ConfirmationStrength",
    "DestructiveClass",
    "IdempotencyPolicy",
    "MutationActuator",
    "MutationAuthorization",
    "MutationPreview",
    "MutationPrincipal",
    "MutationPlan",
    "MutationReceipt",
    "MutationTarget",
    "MutationTargetStatus",
    "MutationTransactionError",
    "OperationExecutor",
    "PlanStaleError",
    "RecoveryPolicy",
    "Surface",
    "SurfaceDeniedError",
    "TargetAuthorityPolicy",
    "TargetDurability",
    "TokenConsumedError",
    "TokenExpiredError",
    "build_plan",
    "build_typed_plan",
    "compute_parameter_digest",
    "compute_plan_hash",
    "compute_target_digest",
    "compute_typed_plan_hash",
    "make_target_ref",
    "validate_mutation_plan_integrity",
]

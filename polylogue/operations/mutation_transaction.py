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
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol, TypeVar, runtime_checkable

#: Destructive/mutating classification. Ordered roughly by blast radius:
#: ``reversible`` writes (tags/metadata) can be undone by another write;
#: ``reset`` tombstones rebuildable rows while preserving durable evidence;
#: ``delete`` permanently removes archive rows but re-ingest of the original
#: source can resurrect them; ``excise`` is the durable, cross-tier,
#: re-ingest-proof removal (right-to-forget).
DestructiveClass = Literal["reversible", "reset", "delete", "excise"]

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


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


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
        }


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

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_hash": self.plan_hash,
            "actor": self.actor,
            "role": self.role,
            "capability": self.capability,
            "confirmation_strength": self.confirmation_strength,
            "authorized_at": self.authorized_at,
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

    def prepare(self, actuator: MutationActuator[ArgsT], args: ArgsT) -> MutationPlan:
        """PREPARE: resolve exact targets from live state. Never mutates."""

        return actuator.prepare(args)

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


def make_target_ref(kind: Literal["session", "message", "block", "source"], value: object) -> str:
    """Return a stable ``kind:value`` target ref, the shared vocabulary for plans/receipts."""

    return f"{kind}:{value}"


__all__ = [
    "AuthorizationMismatchError",
    "ConfirmationRequiredError",
    "ConfirmationStrength",
    "DestructiveClass",
    "MutationActuator",
    "MutationAuthorization",
    "MutationPlan",
    "MutationReceipt",
    "MutationTargetStatus",
    "MutationTransactionError",
    "OperationExecutor",
    "PlanStaleError",
    "build_plan",
    "compute_plan_hash",
    "make_target_ref",
]

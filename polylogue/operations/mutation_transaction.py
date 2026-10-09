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

import asyncio
import hashlib
import json
import os
import secrets
import threading
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeVar, cast, runtime_checkable

from polylogue.core.enums import PrincipalSurface
from polylogue.core.errors import SchemaRefusalError
from polylogue.core.sqlite_locking import is_transient_sqlite_lock
from polylogue.operations.machine_receipts import MachineHistoricalReceipt, encode_machine_receipt

if TYPE_CHECKING:
    from polylogue.operations.audit import AcceptedIdentityResetCustody, AuditRepository
    from polylogue.operations.bindings import OperationBinding
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

#: Destructive/mutating classification. Ordered roughly by blast radius:
#: ``reversible`` writes (tags/metadata) can be undone by another write;
#: ``reset`` tombstones rebuildable rows while preserving durable evidence;
#: ``delete`` permanently removes archive rows but re-ingest of the original
#: source can resurrect them; ``excise`` is the durable, cross-tier,
#: re-ingest-proof removal (right-to-forget); ``maintenance`` rewrites
#: rebuildable derived state without removing authored evidence.
DestructiveClass = Literal["additive", "reversible", "maintenance", "reset", "delete", "excise"]

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
#: How startup recovery resolved one interrupted operation; see
#: :class:`RecoveryResolution`. There is no ``unknown``: every current
#: actuator either re-applies convergently or commits atomically.
RecoveryOutcome = Literal["complete", "absent", "not-replayable", "replay-failed"]


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


class RecoveryBlockedError(MutationTransactionError):
    """A nonterminal overlapping effect cannot be proved safe to overlap."""


@dataclass(frozen=True, slots=True)
class MutationPrincipal:
    """Authenticated identity and exact capabilities for one mutation request."""

    actor_ref: str
    capabilities: frozenset[str]
    surface: PrincipalSurface
    role_label: str | None = None

    def __post_init__(self) -> None:
        if not self.actor_ref:
            raise ValueError("mutation principal actor_ref must not be empty")
        if any(not capability for capability in self.capabilities):
            raise ValueError("mutation principal capabilities must not contain empty values")

    def on_surface(self, surface: PrincipalSurface) -> MutationPrincipal:
        """Return this identity acting through the public surface that owns an operation.

        The daemon derives a principal from its transport, and every machine
        protocol peer is labelled ``cli``. An operation family that belongs to
        one public surface -- the ``user.*`` overlay and ``mutation.facade.*``
        products are the Python/HTTP API's, with no CLI route -- executes under
        that surface, so its actuators' surface allowlists are checked against
        the surface the operation serves rather than the socket it arrived on.
        """
        return replace(self, surface=surface)


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


#: Daemon publication pages bound resident work without limiting a plan's
#: total authorized target set.
MUTATION_PLAN_PAGE_SIZE = 256

#: How many canonical session IDs a delete preview result names. The selection
#: itself has no count cap, so the result reports its size and a leading sample
#: instead of echoing every ID past the operation result bound; the full
#: selection stays in the durable preview chunks the result's refs name.
DELETE_PREVIEW_SAMPLE_IDS = 20


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
    surface: PrincipalSurface | None = None

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
    # Generic domain data never becomes audit history.  This closed field is
    # the opt-in checkpoint for machine operations only.
    historical_receipt: MachineHistoricalReceipt | None = None
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
            "historical_receipt": (
                None if self.historical_receipt is None else encode_machine_receipt(self.historical_receipt)
            ),
            "operation_id": self.operation_id,
        }


@dataclass(frozen=True, slots=True)
class StartedBoundMutation:
    """Durable intent produced by the shared bound-execution preflight.

    The daemon may release its writer lease after :meth:`OperationExecutor.begin_bound`
    and before it obtains a domain result.  This object deliberately contains
    only the authenticated plan and the durable operation reference; it is not
    an authority to re-resolve targets or invoke an actuator on its own.
    """

    plan: MutationPlan
    authorization: MutationAuthorization
    operation_id: str | None


@dataclass(slots=True)
class _StartedMutationTransfer:
    actuator: object
    started: StartedBoundMutation
    pid: int
    thread: threading.Thread
    task: object | None
    consumed: bool = False


_ACTIVE_STARTED_MUTATION: ContextVar[_StartedMutationTransfer | None] = ContextVar(
    "operation_active_started_mutation", default=None
)


def _mutation_task() -> object | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def take_started_bound_mutation(actuator: object, plan: MutationPlan) -> StartedBoundMutation:
    """Transfer this executor's actual begun carrier once to its participant.

    Context copies share the consumed frame. An inherited callback cannot
    replay it after the original participant has taken the carrier.
    """
    transfer = _ACTIVE_STARTED_MUTATION.get()
    if (
        transfer is None
        or transfer.consumed
        or transfer.actuator is not actuator
        or transfer.started.plan is not plan
        or transfer.pid != os.getpid()
        or transfer.thread is not threading.current_thread()
        or transfer.task is not _mutation_task()
    ):
        raise MutationTransactionError("started mutation transfer does not belong to this exact participant")
    transfer.consumed = True
    _ACTIVE_STARTED_MUTATION.set(None)
    return transfer.started


@dataclass(frozen=True, slots=True)
class RecoveryResolution:
    """How one interrupted operation was resolved from durable state.

    ``complete``
        The plan's effect is present: re-applied convergently, or an atomic
        apply's commit was found. ``receipt`` is the replay's receipt.
    ``absent``
        An atomic apply never committed; nothing of it is present.
    ``not-replayable``
        No current actuator declares this operation family and version.
    ``replay-failed``
        Re-applying the plan raised or refused; ``detail`` names why.

    Every outcome is terminal and none is a barrier over later mutations of
    the same targets: an operator re-issuing the request re-applies it through
    the same convergent route.
    """

    outcome: RecoveryOutcome
    detail: str
    receipt: MutationReceipt | None = None

    def __post_init__(self) -> None:
        if (self.outcome == "complete") != (self.receipt is not None):
            raise ValueError("only a complete recovery carries a replay receipt")
        if self.receipt is not None and self.receipt.status not in {"applied", "already_satisfied"}:
            raise ValueError("a complete recovery receipt must report its effect present")


class RecoveryDeferredError(MutationTransactionError):
    """A recovery needs archive state that has not converged yet; retry later."""


class RecoveryRedrivenByOwnerError(MutationTransactionError):
    """A resident owner re-drives this interrupted operation to its own terminal receipt.

    Generic recovery leaves the run nonterminal and does not treat it as a
    barrier: its targets are private to the interrupted operation, and only
    the owner can finish it. Startup recovery and a later request's executor
    both skip it.
    """


class RecoverySettledIndeterminateError(RecoveryRedrivenByOwnerError):
    """A stopped operation with a possibly partial effect keeps its indeterminate state.

    Its outcome is decided (it was stopped) but not absent: generic recovery
    must not rewrite it as failed with no effect, so it is skipped like an
    owner-driven run.
    """


class ReplayHandles:
    """Writable handles recovery gives an actuator to resolve one plan.

    The archive store is opened on first use: most actuators resolve from
    ``archive_root`` alone, and opening it validates every tier, so an eager
    open would refuse recovery for a derived tier convergence is about to
    replace.
    """

    def __init__(self, archive_root: Path, *, input_demand: Callable[[int], None] | None = None) -> None:
        self.archive_root = archive_root
        self.input_demand = input_demand
        self.recovery_operation: RecoveryOperation | None = None
        self._archive: ArchiveStore | None = None

    @property
    def archive(self) -> ArchiveStore:
        if self._archive is None:
            from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

            self._archive = ArchiveStore.open_existing(self.archive_root, read_only=False)
        return self._archive

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None


@dataclass(frozen=True, slots=True)
class RecoveryOperation:
    """Immutable audit evidence handed to a domain-owned recovery inspector."""

    operation_id: str
    operation: str
    operation_version: int
    plan_hash: str
    target_digest: str
    targets: tuple[MutationTarget, ...]
    expected_target_count: int
    reconstructed_target_count: int
    target_evidence_complete: bool
    context: Mapping[str, object] = field(default_factory=dict)
    attempt_id: str | None = None

    @property
    def target_evidence_detail(self) -> str:
        return (
            f"reconstructed {self.reconstructed_target_count} of {self.expected_target_count} durable recovery targets"
        )


class ConvergentReplay:
    """Recovery contract for an actuator whose ``apply`` converges when re-run.

    ``apply(plan, replay_args(handles, plan))`` must reach the plan's effect
    from any state an interrupted apply of the same plan can leave -- nothing
    applied, some targets applied, or everything applied -- and report
    ``applied`` or ``already_satisfied``. The plan context therefore carries
    every input ``apply`` reads; ``replay_args`` rebuilds the arguments from it
    and the writable handles, never from caller state that did not survive.
    """

    #: Whether a replay unlinks or replaces archive files, so a handle opened
    #: before the recovery keeps reading and writing the discarded file.
    replaces_archive_files: ClassVar[bool] = False

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> Any:
        raise NotImplementedError(f"{type(self).__name__} declares no replay arguments")

    def already_applied(self, handles: ReplayHandles, plan: MutationPlan) -> bool:
        """Whether the plan's exact effect is already stored.

        An upsert that restamps its row on every write overrides this, so a
        replay after a committed apply leaves the stored row untouched.
        """
        return False

    def replay_refusal(self, handles: ReplayHandles, plan: MutationPlan) -> str | None:
        """Why re-applying now would act outside the authorized plan, if it would."""
        return None

    def recover(self, handles: ReplayHandles, plan: MutationPlan) -> RecoveryResolution:
        if self.already_applied(handles, plan):
            return RecoveryResolution(
                "complete",
                "the interrupted plan's effect is already stored",
                MutationReceipt(
                    operation=plan.operation,
                    plan_hash=plan.plan_hash,
                    status="already_satisfied",
                    target_refs=plan.target_refs,
                    affected_count=0,
                    detail="effect_already_stored",
                    receipt_ref=None,
                    applied_at=plan.prepared_at,
                ),
            )
        refusal = self.replay_refusal(handles, plan)
        if refusal is not None:
            return RecoveryResolution("replay-failed", refusal)
        try:
            receipt = cast(MutationReceipt, cast(Any, self).apply(plan, self.replay_args(handles, plan)))
        except KeyError as exc:
            # User-tier writers resolve their session through the rebuildable
            # index; a session it cannot resolve yet (an index still
            # converging) is not evidence the write can never land.
            raise RecoveryDeferredError(f"{plan.operation} target does not resolve yet: {exc}") from exc
        if receipt.status in {"applied", "already_satisfied"}:
            return RecoveryResolution("complete", "re-applied the interrupted plan convergently", receipt)
        return RecoveryResolution("replay-failed", f"re-applying the plan ended {receipt.status}: {receipt.detail}")


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
        # The audit tier is authoritative in production.  Keep the same
        # binding locally for daemonless/library executors so a caller cannot
        # forge a token-bearing dataclass after AUTHORIZE.
        self._issued_authorizations: dict[str, MutationAuthorization] = {}
        self._prevalidated_executions: ContextVar[tuple[tuple[object, StartedBoundMutation], ...]] = ContextVar(
            "operation_executor_prevalidated_executions", default=()
        )

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
        from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation

        raw_plan = binding.actuator.prepare(args)
        return self.prepare_bound(
            binding,
            args,
            principal,
            archive_instance_id=self._audit.ensure_archive_authority(now_ms=self._now_ms()),
            archive_identity_digest=ArchiveIdentity.resolve_location(
                ArchiveLocation.resolve(archive_root)
            ).authority_identity_digest,
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
        identity_reset_custody: AcceptedIdentityResetCustody | None = None,
    ) -> MutationAuthorization:
        """Issue a random one-time token bound to the persisted preview."""

        binding.validate()
        validate_mutation_plan_integrity(preview.plan)
        plan = preview.plan
        required = set(plan.required_capabilities)
        if not required.issubset(principal.capabilities):
            missing = sorted(required - principal.capabilities)
            raise CapabilityDeniedError(f"principal lacks declared capabilities: {missing}")
        if identity_reset_custody is not None:
            identity_reset_custody.require_authorization(preview, principal)
        progress_owned = (
            self._audit is not None and self._audit.insight_expiry_policy(preview, principal) == "maintenance_progress"
        )
        if identity_reset_custody is None and not progress_owned and self._now_ms() >= plan.expires_at_ms:
            raise TokenExpiredError("cannot authorize an expired preview")
        # A destructive plan is never authorized with the interim boolean
        # strength.  Callers that omit the strength receive the canonical
        # bound token; an explicit weaker strength remains an actionable
        # rejection so adapters cannot silently downgrade the contract.
        strength = confirmation_strength or (
            "bound_token" if plan.destructive_class in {"reset", "delete", "excise"} else plan.required_confirmation
        )
        if _STRENGTH_ORDER[strength] < _STRENGTH_ORDER[plan.required_confirmation]:
            raise ConfirmationRequiredError(
                f"{binding.spec.name!r} requires {plan.required_confirmation!r}, got {strength!r}"
            )
        if identity_reset_custody is not None and strength != "bound_token":
            raise ConfirmationRequiredError("accepted identity reset requires bound confirmation")
        token = self._token_factory()
        issued_at_ms = self._now_ms()
        authorization = MutationAuthorization(
            plan_hash=plan.plan_hash,
            actor=principal.actor_ref,
            role=principal.role_label or "",
            capability=plan.required_capabilities[0] if plan.required_capabilities else "",
            confirmation_strength=strength,
            authorized_at=_utcnow_iso(),
            preview_ref=preview.preview_ref,
            token=token,
            expires_at_ms=(
                identity_reset_custody.issued_at_ms
                if identity_reset_custody is not None
                else issued_at_ms
                if progress_owned
                else plan.expires_at_ms
            ),
            capabilities=tuple(sorted(required)),
            surface=principal.surface,
        )
        if self._audit is not None:
            authorization_id = self._audit.issue_authorization(
                preview, principal, authorization, issued_at_ms=issued_at_ms
            )
            authorization = replace(authorization, authorization_id=authorization_id)
        assert authorization.token is not None
        self._issued_authorizations[authorization.token] = authorization
        return authorization

    def execute_bound(
        self,
        binding: OperationBinding[ArgsT, object],
        preview: MutationPreview,
        authorization: MutationAuthorization,
        args: ArgsT,
    ) -> MutationReceipt:
        """Consume a bound token, journal intent, apply, and finalize honestly."""

        from polylogue.core.stage_admission import admit_stage_write

        if self._audit is not None:
            self._resolve_dead_operations(
                resolver_actor_ref=authorization.actor,
                prepared_excision_only=True,
                input_demand=getattr(args, "input_demand", None),
            )
        started = admit_stage_write(
            f"operation.{binding.spec.name}.begin", lambda: self.begin_bound(binding, preview, authorization, args)
        )
        active_executions = self._prevalidated_executions.get()
        scope_token = self._prevalidated_executions.set((*active_executions, (binding.actuator, started)))
        try:
            receipt = self.execute(binding.actuator, started.plan, authorization, args)
        except BaseException as exc:
            error_summary = str(exc)[:512]
            from polylogue.core.errors import RefusedBeforeEffectError

            # Excision commits Source completion before its Index mutation,
            # so an Index-side refusal cannot prove the whole operation
            # without effect; it stays indeterminate below.
            if isinstance(exc, RefusedBeforeEffectError) and started.plan.operation != "mutate-session-excision":
                refused = MutationReceipt(
                    operation=started.plan.operation,
                    plan_hash=started.plan.plan_hash,
                    status="blocked",
                    target_refs=started.plan.target_refs,
                    affected_count=0,
                    detail=error_summary,
                    receipt_ref=None,
                    applied_at=_utcnow_iso(),
                )
                try:
                    admit_stage_write(
                        f"operation.{binding.spec.name}.refused",
                        lambda: self.finalize_bound(started, receipt=refused, error_summary=error_summary),
                    )
                except BaseException as cleanup:
                    raise BaseExceptionGroup("Mutation refusal and its finalization failed", [exc, cleanup]) from exc
                raise

            def finalize_indeterminate() -> None:
                # Source completion is already durable if an Excision fault
                # followed that commit. Project the same pending command
                # before appending the attempt's indeterminate transition.
                # Failed physical settlement still refuses this admission.
                if self._audit is not None and started.plan.operation == "mutate-session-excision":
                    self._audit.reconcile_continuity()
                self.finalize_bound(
                    started,
                    error_summary=error_summary,
                    unknown_reason="actuator exception after durable intent",
                )

            try:
                admit_stage_write(
                    f"operation.{binding.spec.name}.indeterminate",
                    finalize_indeterminate,
                )
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Mutation apply and indeterminate finalization failed", [exc, cleanup]
                ) from exc
            raise
        finally:
            self._prevalidated_executions.reset(scope_token)
        completed = admit_stage_write(
            f"operation.{binding.spec.name}.finalize", lambda: self.finalize_bound(started, receipt=receipt)
        )
        assert completed is not None
        return completed

    def begin_bound(
        self,
        binding: OperationBinding[ArgsT, object],
        preview: MutationPreview,
        authorization: MutationAuthorization,
        args: ArgsT,
    ) -> StartedBoundMutation:
        """Validate a bound execution and record durable intent before domain work.

        This is the first of the reusable execution phases.  It performs the
        same fresh-plan, archive-identity, principal, expiry, overlap, and
        one-shot checks that :meth:`execute_bound` historically performed.
        Callers that later await a domain owner must use the returned exact
        plan; they must not call ``prepare`` again after this point.
        """

        binding.validate()
        if binding.spec.name == "mutate-rebuild-insights":
            # A rebuild-insights preview may be an immutable, replayable
            # staging page, but it has no execution authority until the
            # complete page chain is sealed into exact machine parts.  Letting
            # the generic one-plan path consume it would make
            # ``stage_preview -> authorize -> execute_bound`` a bypass around
            # the source-WAL seal.  This family starts only through
            # ``begin_accepted_insight_part`` under a bound machine ordinal.
            raise MutationTransactionError(
                "insight maintenance requires a sealed accepted machine part; use begin_accepted_insight_part"
            )
        validate_mutation_plan_integrity(preview.plan)
        if authorization.preview_ref != preview.preview_ref or (
            authorization.token is None and (self._audit is None or authorization.authorization_id is None)
        ):
            raise AuthorizationMismatchError("authorization is not bound to this preview")
        token = authorization.token or ""

        def revoke_local() -> None:
            self._issued_authorizations.pop(token, None)

        issued = self._issued_authorizations.get(authorization.token or "")
        if self._audit is None and (issued is None or issued != authorization):
            raise AuthorizationMismatchError("authorization token was not issued for this executor")
        if (
            preview.plan.destructive_class in {"reset", "delete", "excise"}
            and authorization.confirmation_strength != "bound_token"
        ):
            raise ConfirmationRequiredError(
                f"{binding.spec.name!r} destructive execution requires a bound preview token"
            )
        if (
            self._audit is None
            and authorization.expires_at_ms is not None
            and self._now_ms() >= authorization.expires_at_ms
        ):
            raise TokenExpiredError("authorization token is expired")
        if self._archive_root is not None:
            from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation

            live_identity = ArchiveIdentity.resolve_location(
                ArchiveLocation.resolve(self._archive_root)
            ).authority_identity_digest
            if live_identity != preview.plan.archive_identity_digest:
                revoke_local()
                raise PlanStaleError("archive identity changed after the bound preview was prepared")
        if self._audit is not None:
            # Land interrupted work before re-preparing, so the freshness
            # check below compares against the state that work leaves.
            self._resolve_dead_operations(resolver_actor_ref=authorization.actor)
        fresh_plan = self._typed_plan_from_actuator(
            binding,
            binding.actuator.prepare(args),
            archive_instance_id=preview.plan.archive_instance_id,
            archive_identity_digest=preview.plan.archive_identity_digest,
            parameter_digest=preview.plan.parameter_digest,
            expires_at_ms=preview.plan.expires_at_ms,
        )
        if fresh_plan.plan_hash != preview.plan.plan_hash:
            revoke_local()
            if self._audit is not None:
                self._audit.mark_preview_stale(preview)
            raise PlanStaleError(
                f"{binding.spec.name!r} preview {preview.plan.plan_hash!r} is stale; "
                f"live state now resolves to {fresh_plan.plan_hash!r}"
            )
        # Tokens are one-shot even for daemonless/library executors. Durable
        # audit rows enforce this in production; this local consume closes
        # the equivalent replay path when no audit repository is configured.
        revoke_local()
        if self._audit is not None:
            self._refuse_unresolved_overlap(fresh_plan)
        operation_id: str | None = None
        if self._audit is not None:
            operation_id = self._audit.consume_authorization_and_start(preview, authorization)
        return StartedBoundMutation(plan=preview.plan, authorization=authorization, operation_id=operation_id)

    def finalize_bound(
        self,
        started: StartedBoundMutation,
        *,
        receipt: MutationReceipt | None = None,
        error_summary: str | None = None,
        unknown_reason: str | None = None,
    ) -> MutationReceipt | None:
        """Finalize one previously begun execution without inventing an effect.

        A successful domain result must be exactly bound to the plan recorded
        by :meth:`begin_bound`.  Supplying no receipt records an indeterminate
        attempt, which is the only truthful outcome for a post-intent failure
        whose domain commit cannot be established.
        """

        if (receipt is None) == (unknown_reason is None):
            raise ValueError("finalization requires exactly one of receipt or unknown_reason")
        if receipt is not None and (
            receipt.operation != started.plan.operation
            or receipt.plan_hash != started.plan.plan_hash
            or receipt.target_refs != started.plan.target_refs
        ):
            raise AuthorizationMismatchError("domain receipt does not match the started exact plan")
        if self._audit is None or started.operation_id is None:
            return receipt
        try:
            if receipt is None:
                self._audit.finalize_attempt(
                    started.operation_id,
                    status="unknown",
                    error_summary=error_summary,
                    unknown_reason=unknown_reason,
                )
                return None
            self._audit.finalize_attempt(
                started.operation_id, status=receipt.status, receipt=receipt, error_summary=error_summary
            )
        except Exception as exc:
            if receipt is None:
                raise
            raise AuditFinalizationError("domain effect is not reported completed without audit finalization") from exc
        return replace(
            receipt,
            receipt_ref=f"mutation-operation:{started.operation_id}",
            operation_id=started.operation_id,
        )

    def begin_accepted_insight_part(
        self,
        binding: OperationBinding[ArgsT, object],
        accepted: object,
        *,
        principal: MutationPrincipal,
        machine_part: int,
        args: ArgsT,
    ) -> StartedBoundMutation:
        """Start one sealed insight page without reopening its accepted scope.

        This is intentionally narrower than :meth:`begin_bound`: it is only
        usable when an existing audit machine part has reserved the exact
        authorization, and it reloads the immutable plan from that authority.
        It therefore cannot be used by a public caller to turn ``None`` or a
        fresh archive scan into an accepted full sweep after the fact.
        """

        from polylogue.operations.insight_acceptance import AcceptedInsightPart, accepted_part_from_plan

        if not isinstance(accepted, AcceptedInsightPart):
            raise TypeError("accepted insight execution requires a sealed AcceptedInsightPart")
        if self._audit is None:
            raise MutationTransactionError("accepted insight execution requires durable audit authority")
        binding.validate()
        preview, authorization = self._audit.authorization_for_principal(accepted.authorization_ref, principal)
        durable = accepted_part_from_plan(
            preview.plan,
            preview_ref=preview.preview_ref,
            authorization_ref=accepted.authorization_ref,
        )
        if durable != accepted:
            raise AuthorizationMismatchError("accepted insight part differs from its durable preview authority")
        if accepted.ordinal != machine_part:
            raise AuthorizationMismatchError("accepted insight page ordinal differs from its machine reservation")
        if preview.plan.operation != binding.spec.name or authorization.plan_hash != preview.plan.plan_hash:
            raise AuthorizationMismatchError("accepted insight authority does not match this operation binding")
        if self._archive_root is not None:
            from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation

            live_identity = ArchiveIdentity.resolve_location(
                ArchiveLocation.resolve(self._archive_root)
            ).authority_identity_digest
            if live_identity != preview.plan.archive_identity_digest:
                raise PlanStaleError("archive identity changed after the insight manifest was accepted")
        self._resolve_dead_operations(resolver_actor_ref=authorization.actor)
        self._refuse_unresolved_overlap(preview.plan)
        operation_id = self._audit.consume_authorization_and_start(preview, authorization)
        return StartedBoundMutation(plan=preview.plan, authorization=authorization, operation_id=operation_id)

    def _resolve_dead_operations(
        self,
        *,
        resolver_actor_ref: str,
        prepared_excision_only: bool = False,
        input_demand: Callable[[int], None] | None = None,
    ) -> None:
        """Land every dead interrupted operation this process can route.

        All of them, not only those sharing the new request's targets: a plan
        derived from live rows cannot name a target an interrupted write had
        not created yet, and that write must land first. A family whose actuator
        module this process never imported stays for daemon startup, and
        refuses only a request whose targets it overlaps.
        """

        assert self._audit is not None
        # Discovery is a settled continuity read on the original creator.
        # Opening a writable audit handle here would bind an unbound outer
        # lease before the admitted BEGIN tries to acquire the same custody.
        with self._audit.recovery_discovery_read():
            orphaned = self._audit.orphaned_operations()
            # Target overlap cannot fence work that deletes archive files: a write
            # to logically unrelated rows still lands in a database the unrouted
            # reset will unlink once startup recovery replays it.
            for operation in orphaned:
                if operation.operation in _RECOVERY_ROUTES:
                    continue
                if any(
                    ref.startswith("path:") for ref in self._audit.operation_plan(operation.operation_id).target_refs
                ):
                    raise RecoveryBlockedError(
                        f"interrupted {operation.operation} {operation.operation_id!r} replaces archive files and "
                        "awaits daemon startup recovery; restart polylogued"
                    )
        dead = tuple(operation for operation in orphaned if operation.operation in _RECOVERY_ROUTES)
        if prepared_excision_only:
            dead = tuple(operation for operation in dead if operation.operation == "mutate-session-excision")
        elif any(operation.operation == "mutate-session-excision" for operation in dead):
            raise RecoveryBlockedError(
                "interrupted Excision requires the original prepared recovery phase before begin"
            )
        if not dead:
            return
        deferred = resolve_interrupted_operations(
            self._audit,
            self._audit.path.parent,
            dead,
            resolver_actor_ref=resolver_actor_ref,
            input_demand=input_demand,
        )
        if deferred:
            raise RecoveryBlockedError(
                f"interrupted operation {deferred[0]!r} awaits archive state this runtime cannot resolve yet; "
                "retry after convergence"
            )
        # The caller opened its archive handles before this recovery ran; a
        # recovered file reset leaves them on unlinked files, so the request
        # must be reissued against fresh ones.
        replaced = next(
            (
                operation
                for operation in dead
                if getattr(_RECOVERY_ROUTES[operation.operation], "replaces_archive_files", False)
            ),
            None,
        )
        if replaced is not None:
            raise RecoveryBlockedError(
                f"recovered interrupted {replaced.operation} {replaced.operation_id!r}; "
                "retry so the request opens the archive files it left"
            )

    def _refuse_unresolved_overlap(self, plan: MutationPlan) -> None:
        """Refuse while unfinished work still holds any of these targets.

        After :meth:`_resolve_dead_operations` the only nonterminal runs left
        are live owners (concurrent work) or families this process cannot
        route; either must finish first.
        """

        assert self._audit is not None
        for operation in self._audit.nonterminal_operations_overlapping(plan.target_refs):
            liveness = self._audit.attempt_owner_liveness(operation.operation_id)
            if liveness != "dead":
                raise RecoveryBlockedError(f"overlapping operation {operation.operation_id!r} has a {liveness} owner")
            raise RecoveryBlockedError(
                f"interrupted {operation.operation} work awaits daemon startup recovery; restart polylogued"
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

        if actuator.operation == "mutate-rebuild-insights":
            raise MutationTransactionError("insight maintenance runs only through a sealed accepted machine part owner")
        if (
            plan.targets
            and plan.destructive_class in {"reset", "delete", "excise"}
            and (
                (authorization.token is None and not self._prevalidated_executions.get())
                or authorization.confirmation_strength != "bound_token"
            )
        ):
            raise ConfirmationRequiredError("destructive execution requires a bound preview token")
        if authorization.plan_hash != plan.plan_hash:
            raise AuthorizationMismatchError(
                f"authorization bound to plan {authorization.plan_hash!r} does not match plan {plan.plan_hash!r}"
            )
        active_executions = self._prevalidated_executions.get()
        if active_executions:
            expected_actuator, started = active_executions[-1]
            if actuator is not expected_actuator or plan is not started.plan:
                raise MutationTransactionError("prevalidated execution does not match the active bound mutation")
            # Context variables are copied into callbacks/tasks at creation.
            # Remove this one-shot authority before entering actuator-owned
            # code so work it schedules cannot inherit and replay it later.
            cleared = self._prevalidated_executions.set(())
            try:
                removal_scope = (
                    nullcontext()
                    if plan.operation == "mutate-session-excision"
                    else _authorized_removal_apply(plan, self._archive_root, actuator, args)
                )
                with removal_scope:
                    transfer = _StartedMutationTransfer(
                        actuator, started, os.getpid(), threading.current_thread(), _mutation_task()
                    )
                    transfer_token = _ACTIVE_STARTED_MUTATION.set(transfer)
                    try:
                        return actuator.apply(plan, args)
                    finally:
                        transfer.consumed = True
                        _ACTIVE_STARTED_MUTATION.reset(transfer_token)
            finally:
                self._prevalidated_executions.reset(cleared)
        fresh_plan = actuator.prepare(args)
        if fresh_plan.plan_hash != plan.plan_hash:
            raise PlanStaleError(
                f"{actuator.operation!r} plan {plan.plan_hash!r} is stale; "
                f"live state now resolves to {fresh_plan.plan_hash!r} "
                f"({fresh_plan.target_count} target(s) vs {plan.target_count})"
            )
        with _authorized_removal_apply(plan, self._archive_root, actuator, args):
            return actuator.apply(plan, args)


@contextmanager
def _authorized_removal_apply(
    plan: MutationPlan, archive_root: Path | None, actuator: object, args: object | None = None
) -> Iterator[None]:
    # Only these registered actuators intentionally remove stored sessions.
    # Other operations may name sessions without authorizing their absence.
    if plan.operation not in {"mutate-delete-session", "mutate-identity-reset", "mutate-session-excision"}:
        yield
        return
    registered = _RECOVERY_ROUTES.get(plan.operation)
    if registered is None or type(actuator) is not type(registered):
        raise MutationTransactionError("session disappearance requires the registered removal actuator")
    if archive_root is None:
        # The registered domain actuator owns this exact argument shape.
        # Do not import domain implementations into the transaction authority.
        if args is None:
            raise MutationTransactionError("session removal has no declared archive destination")
        if plan.operation == "mutate-delete-session":
            archive_root = cast(Any, args).archive._write_lease_archive_root
        else:
            archive_root = cast(Any, args).archive_root
    from polylogue.core.write_lease import authorized_session_removal, write_lease

    session_ids = tuple(ref.removeprefix("session:") for ref in plan.target_refs if ref.startswith("session:"))
    # Library execution acquires the same existing root-bound lease; admitted
    # daemon execution borrows that exact physical owner on its creator.
    with (
        write_lease("operation.authorized-removal", archive_root=archive_root),
        authorized_session_removal(
            archive_root=archive_root,
            plan_hash=plan.plan_hash,
            session_ids=session_ids,
            excise_assertions=plan.operation == "mutate-session-excision",
        ),
    ):
        yield


class RecoverableActuator(Protocol):
    """An actuator that resolves its own interrupted plan from durable state."""

    @property
    def operation(self) -> str: ...

    def recover(self, handles: ReplayHandles, plan: MutationPlan) -> RecoveryResolution: ...


#: Recovery routes by operation name. Each actuator module registers its own
#: families when imported, so this module never imports the actuators and
#: stays out of their dependency graph; ``mutation_replay`` imports them all.
_RECOVERY_ROUTES: dict[str, RecoverableActuator] = {}


def register_recovery_route(*actuators: RecoverableActuator) -> None:
    """Register the recovery route for each actuator's operation family."""

    for actuator in actuators:
        existing = _RECOVERY_ROUTES.get(actuator.operation)
        if existing is not None and type(existing) is not type(actuator):
            raise ValueError(f"conflicting recovery routes for {actuator.operation!r}")
        _RECOVERY_ROUTES[actuator.operation] = actuator


def registered_recovery_routes() -> dict[str, RecoverableActuator]:
    return dict(_RECOVERY_ROUTES)


def resolve_interrupted_operation(
    audit: AuditRepository, handles: ReplayHandles, operation: RecoveryOperation
) -> RecoveryResolution:
    """Decide one dead operation's outcome from durable state."""

    from polylogue.operations.specs import build_runtime_operation_catalog

    actuator = _RECOVERY_ROUTES.get(operation.operation)
    if actuator is None:
        return RecoveryResolution("not-replayable", f"no current actuator declares {operation.operation!r}")
    version = getattr(actuator, "operation_version", None)
    if version is None:
        spec = build_runtime_operation_catalog().by_name().get(operation.operation)
        if spec is None:
            return RecoveryResolution("not-replayable", f"no current operation spec declares {operation.operation!r}")
        version = spec.operation_version
    if version != operation.operation_version:
        return RecoveryResolution(
            "not-replayable",
            f"{operation.operation!r} v{operation.operation_version} was retired; this runtime declares v{version}",
        )
    if operation.operation == "mutate-resolve-raw-authority-blocker" and handles.input_demand is not None:
        from polylogue.core.stage_admission import admit_stage_write

        plan = admit_stage_write("operation.recovery.plan", lambda: audit.operation_plan(operation.operation_id))
    else:
        with audit.settled_machine_read():
            plan = audit.operation_plan(operation.operation_id)
    if plan.plan_hash != operation.plan_hash or plan.operation != operation.operation:
        raise AuthorizationMismatchError("recovery operation differs from its original durable plan")
    if handles.recovery_operation is not None and handles.recovery_operation is not operation:
        raise AuthorizationMismatchError("recovery handles already belong to another original operation")
    handles.recovery_operation = operation
    try:
        if operation.operation == "mutate-session-excision":
            # Original preparation runs off admission. The domain's exact
            # retained command acquires physical removal custody only at apply.
            return actuator.recover(handles, plan)
        with _authorized_removal_apply(plan, handles.archive_root, actuator):
            return actuator.recover(handles, plan)
    except (SchemaRefusalError, RecoveryDeferredError, RecoveryRedrivenByOwnerError):
        raise
    except Exception as exc:
        if operation.operation == "mutate-session-excision":
            # Source or paid effects may already be durable. A refusal or
            # unsettled resource cannot erase that original attempt's fence.
            raise RecoveryDeferredError(
                f"original Excision recovery refused: {type(exc).__name__}: {exc}"[:512]
            ) from exc
        # Actuators can wrap native errors without erasing their typed cause.
        # Contention cannot settle the effect of the original accepted intent.
        cause: BaseException | None = exc
        seen: set[int] = set()
        while cause is not None and id(cause) not in seen:
            if is_transient_sqlite_lock(cause):
                raise RecoveryDeferredError(f"{operation.operation} recovery encountered SQLite contention") from exc
            seen.add(id(cause))
            cause = cause.__cause__
        return RecoveryResolution("replay-failed", f"{type(exc).__name__}: {exc}"[:512])


def resolve_interrupted_operations(
    audit: AuditRepository,
    archive_root: Path,
    operations: tuple[RecoveryOperation, ...],
    *,
    resolver_actor_ref: str,
    input_demand: Callable[[int], None] | None = None,
) -> tuple[str, ...]:
    """Resolve and terminalize each dead operation; return the ids left pending.

    An operation whose recovery needs a tier this runtime cannot serve yet
    (a derived tier awaiting convergence) is left nonterminal and retried at
    the next startup or overlapping request, never terminalized as failed.
    An operation its resident owner re-drives is left to that owner and is
    neither recorded nor returned as pending.
    """

    if not resolver_actor_ref:
        raise ValueError("recovery resolver actor_ref must not be empty")
    deferred: list[str] = []
    for operation in operations:
        # Fresh handles per operation: one resolution (a filesystem reset)
        # may remove the very tier files a cached store would keep open.
        handles = ReplayHandles(archive_root, input_demand=input_demand)
        try:
            resolution = resolve_interrupted_operation(audit, handles, operation)
        except (SchemaRefusalError, RecoveryDeferredError):
            deferred.append(operation.operation_id)
            continue
        except RecoveryRedrivenByOwnerError:
            continue
        finally:
            handles.close()
        from polylogue.core.stage_admission import admit_stage_write

        def finalize_recovery(
            operation: RecoveryOperation = operation, resolution: RecoveryResolution = resolution
        ) -> None:
            audit.record_recovery_resolution(operation.operation_id, resolution, resolver_actor_ref=resolver_actor_ref)

        admit_stage_write("operation.recovery.finalize", finalize_recovery)
    return tuple(deferred)


def make_target_ref(kind: Literal["session", "message", "block", "source", "index", "path"], value: object) -> str:
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
    "RecoveryBlockedError",
    "RecoveryDeferredError",
    "RecoveryRedrivenByOwnerError",
    "RecoverySettledIndeterminateError",
    "ConvergentReplay",
    "RecoveryOutcome",
    "RecoveryResolution",
    "ReplayHandles",
    "RecoveryOperation",
    "RecoveryPolicy",
    "PrincipalSurface",
    "SurfaceDeniedError",
    "StartedBoundMutation",
    "take_started_bound_mutation",
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
    "RecoverableActuator",
    "register_recovery_route",
    "registered_recovery_routes",
    "resolve_interrupted_operation",
    "resolve_interrupted_operations",
    "validate_mutation_plan_integrity",
]

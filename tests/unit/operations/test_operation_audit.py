from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextvars import Context, copy_context
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from polylogue.operations.audit import (
    AuditRepository,
    MachineRequestBinding,
    MachineRequestConflictError,
    MachineRequestRecoveredError,
    _attempt_owner_is_live,
    _attempt_owner_liveness,
    _current_process_attempt_owner,
    token_sha256,
)
from polylogue.operations.bindings import OperationBinding
from polylogue.operations.ingest_acceptance import INGEST_OPERATION
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.machine_receipts import (
    IngestHistoricalReceiptV2,
    IngestInputHistoricalReceipt,
    IngestInputPageHistoricalReceipt,
    IngestInputRawMemberHistorical,
    IngestInputRawPageHistoricalReceipt,
    IngestInsightPageHistoricalReceipt,
    IngestTerminalSummaryHistorical,
    InsightCertifiedCountsHistorical,
    InsightPartHistoricalReceipt,
    InsightTargetHistoricalReceipt,
    ingest_input_pages_digest,
    ingest_input_raw_pages_digest,
    ingest_insight_pages_digest,
    ingest_session_ids_digest,
)
from polylogue.operations.mutation_transaction import (
    AuditFinalizationError,
    AuthorizationMismatchError,
    CapabilityDeniedError,
    ConfirmationStrength,
    DestructiveClass,
    MutationPlan,
    MutationPrincipal,
    MutationReceipt,
    MutationTransactionError,
    OperationExecutor,
    PlanStaleError,
    RecoveryBlockedError,
    RecoveryResolution,
    ReplayHandles,
    StartedBoundMutation,
    TargetAuthorityPolicy,
    TargetDurability,
    TokenConsumedError,
    TokenExpiredError,
    build_plan,
    take_started_bound_mutation,
)
from polylogue.operations.specs import OperationKind, OperationSpec
from polylogue.storage.sqlite.audit_continuity import (
    AuditContinuityCoordinator,
    AuditContinuityUnknownMutationError,
    AuditMutation,
)
from polylogue.storage.sqlite.audit_leaf import (
    AuditLeafError,
    VerifiedAuditLeaf,
    open_verified_audit_connection,
    open_verified_audit_read_connection,
)
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.operation_recovery import recover_on_admitted_owner


@dataclass
class _Actuator:
    operation: str = "mutate-fixture"
    operation_version: int = 1
    changed: bool = False
    calls: int = 0
    crash: bool = False
    recovery_raises: bool = False
    recoveries: int = 0
    target_refs: tuple[str, ...] = ("session:fixture",)
    effect: str | None = None
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, _args: object) -> MutationPlan:
        targets = ("session:changed",) if self.changed else self.target_refs
        context = {} if self.effect is None else {"effect": self.effect}
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=targets,
            affected_tiers=("user",),
            reversible=True,
            context=context,
        )

    def apply(self, plan: MutationPlan, _args: object) -> MutationReceipt:
        self.calls += 1
        if self.crash:
            raise RuntimeError("simulated actuator crash")
        return MutationReceipt(
            operation=plan.operation,
            plan_hash=plan.plan_hash,
            status="applied",
            target_refs=plan.target_refs,
            affected_count=len(plan.target_refs),
            detail=None,
            receipt_ref=None,
            applied_at="now",
        )

    def recover(self, _handles: ReplayHandles, plan: MutationPlan) -> RecoveryResolution:
        self.recoveries += 1
        if self.recovery_raises:
            raise OSError("synthetic target is unreadable")
        return RecoveryResolution("complete", "fixture replay", self.apply(plan, None))


@dataclass(frozen=True)
class _TypedDomainBatch:
    batch_ref: str
    rows: tuple[str, ...]
    _cached_bytes: bytes = field(init=False, repr=False, default=b"private-cache")


class _TypedDomainOutcome(BaseModel):
    row_ref: str
    status: str


@dataclass
class _TypedReceiptActuator(_Actuator):
    def apply(self, plan: MutationPlan, args: object) -> MutationReceipt:
        receipt = super().apply(plan, args)
        return replace(
            receipt,
            domain_receipt={
                "batch": _TypedDomainBatch("annotation-batch:typed", ("assertion:typed",)),
                "outcomes": (_TypedDomainOutcome(row_ref="assertion:typed", status="imported"),),
            },
        )


@dataclass
class _AlreadySatisfiedActuator(_Actuator):
    def apply(self, plan: MutationPlan, args: object) -> MutationReceipt:
        return replace(super().apply(plan, args), status="already_satisfied", affected_count=0)


@dataclass
class _FailedReceiptActuator(_Actuator):
    def apply(self, plan: MutationPlan, args: object) -> MutationReceipt:
        return replace(super().apply(plan, args), status="failed", affected_count=0)


def _binding(
    actuator: _Actuator,
    *,
    target_durability: TargetDurability = "derived",
    operation_name: str = "mutate-fixture",
) -> OperationBinding[object, object]:
    spec = OperationSpec(
        name=operation_name,
        kind=OperationKind.MAINTENANCE,
        description="fixture",
        mutates_state=True,
        executor_status="executor-routed",
        allowed_surfaces=("internal",),
        target_authority=(
            TargetAuthorityPolicy(
                key="session",
                target_kinds=("session",),
                required_capabilities=("archive.fixture.write",),
                destructive_class="reversible",
                required_confirmation="role_only",
                allowed_durabilities=(target_durability,),
                allowed_recovery=("none",),
            ),
        ),
        affected_tiers=("user",),
    )
    return OperationBinding(spec, actuator)


def _delete_binding(actuator: _Actuator) -> OperationBinding[object, object]:
    """Like ``_binding``, but the target-authority policy's destructive class is ``delete``.

    ``_typed_plan_from_actuator`` derives the *effective* ``destructive_class``
    on the resolved plan from the binding's target-authority policy, not the
    raw plan the actuator returned -- ``_binding``'s policy hardcodes
    ``"reversible"``, so any test exercising delete-specific plan behavior
    must use this instead.
    """

    spec = OperationSpec(
        name="mutate-fixture",
        kind=OperationKind.MAINTENANCE,
        description="fixture",
        mutates_state=True,
        executor_status="executor-routed",
        allowed_surfaces=("internal",),
        target_authority=(
            TargetAuthorityPolicy(
                key="session",
                target_kinds=("session",),
                required_capabilities=("archive.fixture.write",),
                destructive_class="delete",
                required_confirmation="confirm_flag",
                allowed_durabilities=("derived",),
                allowed_recovery=("none",),
            ),
        ),
        affected_tiers=("user",),
    )
    return OperationBinding(spec, actuator)


def _principal() -> MutationPrincipal:
    return MutationPrincipal("actor:test", frozenset({"archive.fixture.write"}), "internal", "system")


def _audit(tmp_path: Path) -> AuditRepository:
    bootstrap_archive_root(tmp_path)
    return AuditRepository.for_archive_root(tmp_path)


@pytest.mark.parametrize("crash_phase", ["after_source_prepare", "after_audit_commit"])
def test_machine_request_and_domain_run_replay_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_phase: str
) -> None:
    """Removing the binding from the prepared command loses the exchange after restart."""

    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "synthetic-private-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:fixture",
        archive_identity_digest="identity:fixture",
        parameter_digest="params:fixture",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    binding = MachineRequestBinding("identity:fixture", "request:fixture", "actor:test", "a" * 64, "mutation.fixture")
    original_phase = AuditContinuityCoordinator._phase

    def crash(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "consume_authorization_and_start" and phase == crash_phase:
            raise RuntimeError("synthetic machine acceptance crash")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", crash)
    with audit.bind_machine_request(binding, transition="consume_authorization_and_start"):
        with pytest.raises(RuntimeError, match="machine acceptance crash"):
            executor.execute_bound(_binding(actuator), preview, authorization, object())
    assert actuator.calls == 0
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)
    recovered = AuditRepository.for_archive_root(tmp_path)
    recovered.reconcile_continuity()
    record = recovered.machine_request(binding)
    assert record is not None
    assert record["artifact_kind"] == "operation"
    operation = recovered.get_operation(str(record["artifact_ref"]))
    assert operation is not None and operation["preview_id"] == preview.preview_ref
    with recovered.bind_machine_request(binding, transition="consume_authorization_and_start"):
        with pytest.raises(MachineRequestRecoveredError):
            recovered.consume_authorization_and_start(preview, authorization)
    with pytest.raises(MachineRequestConflictError):
        recovered.machine_request(replace(binding, fingerprint="b" * 64))
    with pytest.raises(MachineRequestConflictError):
        recovered.machine_request(replace(binding, principal_ref="actor:other"))
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM operation_runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM machine_requests").fetchone()[0] == 1
        assert "synthetic-private-token" not in "\n".join(conn.iterdump())


@pytest.mark.parametrize("crash_phase", ["after_source_prepare", "after_audit_commit", None])
def test_machine_batch_reserves_unstarted_suffix_and_never_replays_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_phase: str | None,
) -> None:
    """Eagerly creating suffix runs or omitting legacy-token reservation checks makes this red."""
    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit)
    previews = tuple(
        executor.prepare_bound(
            _binding(actuator),
            object(),
            _principal(),
            archive_instance_id="archive:fixture",
            archive_identity_digest="identity:fixture",
            parameter_digest=f"params:{i}",
        )
        for i in range(2)
    )
    authorizations = tuple(executor.authorize_bound(_binding(actuator), preview, _principal()) for preview in previews)
    refs = tuple(str(auth.authorization_id) for auth in authorizations)
    binding = MachineRequestBinding("identity:fixture", "request:batch", "actor:test", "c" * 64, "mutation.fixture")
    original_phase = AuditContinuityCoordinator._phase

    def crash(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "accept_execution_batch" and phase == crash_phase:
            raise RuntimeError("synthetic batch crash")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", crash)
    with audit.bind_machine_request(binding, transition="accept_execution_batch", deadline_unix_ms=9999999999999):
        if crash_phase:
            with pytest.raises(RuntimeError, match="synthetic batch crash"):
                audit.accept_execution_batch(refs, _principal())
        else:
            audit.accept_execution_batch(refs, _principal())
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)
    audit.reconcile_continuity()
    parts = audit.machine_parts(binding)
    assert [part["authorization_ref"] for part in parts] == list(refs)
    assert all(part["operation_id"] is None for part in parts)
    assert actuator.calls == 0
    with pytest.raises(TokenConsumedError, match="reserved"):
        audit.consume_authorization_and_start(previews[1], authorizations[1])
    with audit.bind_machine_request(binding, transition="consume_authorization_and_start", part=0):
        receipt = executor.execute_bound(_binding(actuator), previews[0], authorizations[0], object())
    assert receipt.affected_count == 1 and actuator.calls == 1
    parts = audit.machine_parts(binding)
    assert parts[0]["operation_id"] is not None and parts[1]["operation_id"] is None
    audit.stop_machine_batch(binding, "cancelled")
    with audit.bind_machine_request(binding, transition="consume_authorization_and_start", part=1):
        with pytest.raises(TokenConsumedError, match="reserved"):
            audit.consume_authorization_and_start(previews[1], authorizations[1])
    recovered = AuditRepository.for_archive_root(tmp_path)
    recovered.reconcile_continuity()
    recovered_request = recovered.machine_request(binding)
    assert recovered_request is not None
    assert recovered_request["stop_reason"] == "cancelled"
    assert recovered_request["accepted_deadline_unix_ms"] == 9999999999999
    assert recovered.machine_parts(binding) == parts
    operation_id = parts[0]["operation_id"]
    assert isinstance(operation_id, str)
    recovered_operation = recovered.get_operation(operation_id)
    assert recovered_operation is not None
    assert recovered_operation["status"] == "completed"
    state = machine_request_state(recovered, recovered_request)
    assert state["outcome"] == "cancelled"
    assert state["completed_chunks"] == 1
    assert state["not_attempted"] == [1]
    assert state["parts"] == [{"ordinal": 0, "operation_id": operation_id, "outcome": "completed", "receipt": None}]
    with recovered.bind_machine_request(binding, transition="accept_execution_batch"):
        with pytest.raises(MachineRequestRecoveredError):
            recovered.accept_execution_batch(refs, _principal())
    assert actuator.calls == 1


@pytest.mark.parametrize("operation_name", ("ingest", "maintenance.insights.rebuild"))
def test_rich_receipt_operation_without_terminal_receipt_stays_indeterminate(
    tmp_path: Path, operation_name: str
) -> None:
    """Removing the rich-receipt guard makes historical ingest/insight replies look completed."""
    audit = _audit(tmp_path)
    actuator = _Actuator(operation=operation_name)
    operation_binding = _binding(actuator, operation_name=operation_name)
    executor = OperationExecutor(audit=audit)
    preview = executor.prepare_bound(
        operation_binding,
        object(),
        _principal(),
        archive_instance_id="archive:receipt-fixture",
        archive_identity_digest="identity:receipt-fixture",
        parameter_digest=f"params:{operation_name}",
    )
    authorization = executor.authorize_bound(operation_binding, preview, _principal())
    assert authorization.authorization_id is not None
    binding = MachineRequestBinding(
        "identity:receipt-fixture",
        f"request:{operation_name}",
        "actor:test",
        "f" * 64,
        operation_name,
    )
    with audit.bind_machine_request(binding, transition="accept_execution_batch"):
        audit.accept_execution_batch((str(authorization.authorization_id),), _principal())
    with audit.bind_machine_request(binding, transition="consume_authorization_and_start", part=0):
        executor.execute_bound(operation_binding, preview, authorization, object())

    record = audit.machine_request(binding)
    assert record is not None
    state = machine_request_state(audit, record)
    assert state["outcome"] == "indeterminate"
    assert state["effect"] == "indeterminate"
    assert state["completed_chunks"] == 1
    assert state["parts"] == [
        {
            "ordinal": 0,
            "operation_id": audit.machine_parts(binding)[0]["operation_id"],
            "outcome": "indeterminate",
            "receipt": None,
        }
    ]


@pytest.mark.parametrize("final_summary", [True, False], ids=["summarized", "unsummarized"])
def test_completed_insight_rebuild_state_carries_its_declared_result(tmp_path: Path, final_summary: bool) -> None:
    """A completed rebuild answers with the summary its final page's receipt closed.

    ``operation.await`` and the executing request both return
    ``state.get("result", state)``; that value must satisfy the operation's
    declared ``InsightRebuildResult``, or the client raises a protocol error
    after the rebuild has already committed.

    Anti-vacuity: remove the insight-rebuild branch from
    ``machine_request_state`` and the summarized case returns the generic
    lifecycle state, which ``validate_operation_result`` refuses; the
    unsummarized case then reports ``completed`` with no result at all.
    """
    from polylogue.operations.daemon_protocol import validate_operation_result
    from polylogue.operations.machine_receipts import InsightTerminalSummaryHistorical

    operation_name = "maintenance.insights.rebuild"
    audit = _audit(tmp_path)
    actuator = _Actuator(operation=operation_name)
    operation_binding = _binding(actuator, operation_name=operation_name)
    executor = OperationExecutor(audit=audit)
    preview = executor.prepare_bound(
        operation_binding,
        object(),
        _principal(),
        archive_instance_id="archive:insight-result",
        archive_identity_digest="identity:insight-result",
        parameter_digest="params:insight-result",
    )
    authorization = executor.authorize_bound(operation_binding, preview, _principal())
    assert authorization.authorization_id is not None
    binding = MachineRequestBinding(
        "identity:insight-result", "request:insight-result", "actor:test", "f" * 64, operation_name
    )
    with audit.bind_machine_request(binding, transition="accept_execution_batch"):
        audit.accept_execution_batch((str(authorization.authorization_id),), _principal())
    with audit.bind_machine_request(binding, transition="consume_authorization_and_start", part=0):
        started = executor.begin_bound(operation_binding, preview, authorization, object())
    history = InsightPartHistoricalReceipt(
        ordinal=0,
        page_count=1,
        manifest_digest="a" * 64,
        index_generation="index-generation:fixture",
        recipe_version="fixture-recipe",
        targets=[
            InsightTargetHistoricalReceipt(
                target_ref="session:fixture",
                disposition="published",
                input_binding="input:fixture",
                output_binding="output:fixture",
                certified_counts=InsightCertifiedCountsHistorical(profiles=1),
                publication_known_committed=True,
            )
        ],
        terminal_summary=(
            InsightTerminalSummaryHistorical(profiles=1, threads=2, tag_rollups=3) if final_summary else None
        ),
    )
    executor.finalize_bound(
        started,
        receipt=MutationReceipt(
            operation=started.plan.operation,
            plan_hash=started.plan.plan_hash,
            status="applied",
            target_refs=started.plan.target_refs,
            affected_count=1,
            detail=None,
            receipt_ref=None,
            applied_at="now",
            historical_receipt=history,
        ),
    )

    record = audit.machine_request(binding)
    assert record is not None
    state = machine_request_state(audit, record)
    validate_operation_result("operation.await", state)
    if final_summary:
        assert state["outcome"] == "completed"
        assert state["result"] == {"profiles": 1, "threads": 2, "tag_rollups": 3}
        validate_operation_result(operation_name, state.get("result", state))
    else:
        assert state["outcome"] == "indeterminate"
        assert "result" not in state


def test_compound_preview_and_authorization_recovery_retains_exact_refs(tmp_path: Path) -> None:
    """Losing any part reference or reserving at authorization creation breaks the next acceptance."""
    audit = _audit(tmp_path)
    executor = OperationExecutor()
    actuator = _Actuator()
    previews = tuple(
        executor.prepare_bound(
            _binding(actuator),
            object(),
            _principal(),
            archive_instance_id="archive:fixture",
            archive_identity_digest="identity:fixture",
            parameter_digest=f"params:{i}",
        )
        for i in range(2)
    )
    binding = MachineRequestBinding("identity:fixture", "request:previews", "actor:test", "d" * 64, "mutation.preview")
    with audit.bind_machine_request(binding, transition="create_preview_batch"):
        refs = audit.create_preview_batch(tuple(preview.plan for preview in previews), _principal())
    assert [part["artifact_ref"] for part in audit.machine_parts(binding)] == refs
    previews = tuple(replace(preview, preview_ref=ref) for preview, ref in zip(previews, refs, strict=True))
    authorizations = tuple(executor.authorize_bound(_binding(actuator), preview, _principal()) for preview in previews)
    auth_binding = replace(binding, request_id="request:authorizations", operation_name="mutation.authorize")
    with audit.bind_machine_request(auth_binding, transition="issue_authorization_batch"):
        auth_refs = audit.issue_authorization_batch(previews, _principal(), authorizations)
    assert [part["artifact_ref"] for part in audit.machine_parts(auth_binding)] == auth_refs
    execute_binding = replace(binding, request_id="request:execute", operation_name="mutation.execute")
    with audit.bind_machine_request(execute_binding, transition="accept_execution_batch"):
        audit.accept_execution_batch(tuple(auth_refs), _principal())
    assert [part["authorization_ref"] for part in audit.machine_parts(execute_binding)] == auth_refs


def test_authenticated_authorization_reference_survives_restart_and_is_one_shot(tmp_path: Path) -> None:
    """Dropping principal/capability checks or consuming a second time must fail."""

    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit)
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:fixture",
        archive_identity_digest="identity:fixture",
        parameter_digest="params:fixture",
    )
    issued = executor.authorize_bound(_binding(actuator), preview, _principal())
    assert issued.authorization_id is not None
    recovered = AuditRepository.for_archive_root(tmp_path)
    for principal in (
        replace(_principal(), actor_ref="actor:other"),
        replace(_principal(), capabilities=frozenset()),
        replace(_principal(), surface="api"),
    ):
        with pytest.raises(AuthorizationMismatchError):
            recovered.authorization_for_principal(issued.authorization_id, principal)
    restored_preview, restored = recovered.authorization_for_principal(issued.authorization_id, _principal())
    assert restored.token is None
    assert restored_preview.plan.plan_hash == preview.plan.plan_hash
    assert restored.expires_at_ms == issued.expires_at_ms
    resumed = OperationExecutor(audit=recovered)
    receipt = resumed.execute_bound(_binding(actuator), restored_preview, restored, object())
    assert receipt.status == "applied" and actuator.calls == 1
    with pytest.raises(TokenConsumedError):
        resumed.execute_bound(_binding(actuator), restored_preview, restored, object())
    assert actuator.calls == 1


def test_token_is_digest_only_and_consumption_run_attempt_are_atomic(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    audit.ensure_archive_authority(now_ms=1)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "raw-secret-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    assert "raw-secret-token" not in (tmp_path / "audit.db").read_bytes().decode("utf-8", errors="ignore")
    digest_as_bearer = replace(authorization, token=f"sha256:{token_sha256('raw-secret-token')}")
    with pytest.raises(AuthorizationMismatchError, match="does not match preview"):
        audit.consume_authorization_and_start(preview, digest_as_bearer)
    receipt = executor.execute_bound(_binding(actuator), preview, authorization, object())
    assert receipt.operation_id is not None
    operation = audit.get_operation(receipt.operation_id)
    assert operation is not None
    assert operation["status"] == "completed"
    assert operation["affected_count"] == 1
    assert audit.list_events(receipt.operation_id)[-1]["event_type"] == "attempt_finalized"
    with pytest.raises(RuntimeError, match="consumed"):
        executor.execute_bound(_binding(actuator), preview, authorization, object())
    assert actuator.calls == 1


def test_execute_bound_routes_the_effect_through_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bound execution reaches the executor's sole actuator.apply gate."""

    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "bound-route-token")
    binding = _binding(actuator)
    preview = executor.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )
    authorization = executor.authorize_bound(binding, preview, _principal())
    calls: list[object] = []
    original_execute = OperationExecutor.execute

    def record_execute(
        self: OperationExecutor, invoked_actuator: object, plan: object, auth: object, args: object
    ) -> object:
        calls.append(invoked_actuator)
        return original_execute(self, invoked_actuator, plan, auth, args)  # type: ignore[arg-type]

    monkeypatch.setattr(OperationExecutor, "execute", record_execute)

    receipt = executor.execute_bound(binding, preview, authorization, object())

    assert receipt.status == "applied"
    assert calls == [actuator]
    assert actuator.calls == 1


@pytest.mark.parametrize("exit_kind", ["success", "exception", "cancellation"])
def test_bound_started_carrier_is_exact_one_shot_and_expires_on_every_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_kind: str
) -> None:
    """The participant borrows the real begin result; copied contexts cannot replay it."""
    begun: list[StartedBoundMutation] = []
    inherited: list[Context] = []
    plans: list[MutationPlan] = []
    original_begin = OperationExecutor.begin_bound

    def record_begin(self: OperationExecutor, *args: Any, **kwargs: Any) -> StartedBoundMutation:
        started = original_begin(self, *args, **kwargs)
        begun.append(started)
        return started

    class Participant(_Actuator):
        def apply(self, plan: MutationPlan, args: object) -> MutationReceipt:
            plans.append(plan)
            inherited.append(copy_context())
            # Equal plans and equal participant instances confer no authority.
            with pytest.raises(MutationTransactionError):
                take_started_bound_mutation(_Actuator(), plan)
            with pytest.raises(MutationTransactionError):
                take_started_bound_mutation(self, replace(plan))
            # A copied context on a foreign thread cannot consume the original.
            foreign_context = copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(foreign_context.run, take_started_bound_mutation, self, plan)
                with pytest.raises(MutationTransactionError):
                    future.result()
            assert take_started_bound_mutation(self, plan) is begun[0]
            assert begun[0].operation_id is not None
            with pytest.raises(MutationTransactionError):
                inherited[0].run(take_started_bound_mutation, self, plan)
            if exit_kind == "exception":
                raise RuntimeError("participant failed after transfer")
            if exit_kind == "cancellation":
                raise asyncio.CancelledError()
            return super().apply(plan, args)

    monkeypatch.setattr(OperationExecutor, "begin_bound", record_begin)
    actuator = Participant()
    executor = OperationExecutor(audit=_audit(tmp_path), token_factory=lambda: "started-carrier-token")
    binding = _binding(actuator)
    preview = executor.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )
    authorization = executor.authorize_bound(binding, preview, _principal())
    if exit_kind == "success":
        receipt = executor.execute_bound(binding, preview, authorization, object())
        assert receipt.operation_id == begun[0].operation_id
    else:
        failure = RuntimeError if exit_kind == "exception" else asyncio.CancelledError
        with pytest.raises(failure):
            executor.execute_bound(binding, preview, authorization, object())
    assert len(begun) == 1
    assert plans == [begun[0].plan]
    with pytest.raises(MutationTransactionError):
        take_started_bound_mutation(actuator, plans[0])
    with pytest.raises(MutationTransactionError):
        inherited[0].run(take_started_bound_mutation, actuator, plans[0])


def test_prepare_bound_uses_the_declared_durable_target_for_legacy_actuators() -> None:
    """Fallback target construction cannot downgrade a durable policy to derived."""

    actuator = _Actuator()
    preview = OperationExecutor().prepare_bound(
        _binding(actuator, target_durability="durable"),
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )

    assert preview.plan.targets[0].durability == "durable"
    assert preview.plan.targets[0].recovery == "none"


def test_production_executor_factory_persists_audit_preview(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor.for_archive_root(tmp_path, token_factory=lambda: "factory-token")

    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT preview_id FROM operation_previews").fetchone()[0] == preview.preview_ref


def test_audit_authority_rejects_a_symlinked_audit_leaf_without_touching_its_target(tmp_path: Path) -> None:
    """Bootstrap and direct audit access never follow an audit path outside its archive root."""

    bootstrap_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    external_audit = tmp_path.parent / "external-audit.db"
    external_audit.write_bytes(audit_path.read_bytes())
    audit_path.unlink()
    audit_path.symlink_to(external_audit)
    before = external_audit.read_bytes()

    with pytest.raises(RuntimeError, match="archive-owned regular file"):
        bootstrap_archive_root(tmp_path)
    with pytest.raises(RuntimeError, match="archive-owned regular file"):
        AuditRepository.for_archive_root(tmp_path).ensure_archive_authority(now_ms=1)

    assert external_audit.read_bytes() == before


def test_audit_authority_rejects_a_hardlinked_audit_leaf_without_touching_its_target(tmp_path: Path) -> None:
    """A regular-looking audit leaf must still have exactly one archive-owned link."""

    bootstrap_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    external_audit = tmp_path.parent / "external-hardlinked-audit.db"
    external_audit.write_bytes(audit_path.read_bytes())
    audit_path.unlink()
    audit_path.hardlink_to(external_audit)
    before = external_audit.read_bytes()

    with pytest.raises(RuntimeError, match="one link"):
        bootstrap_archive_root(tmp_path)
    with pytest.raises(RuntimeError, match="one link"):
        AuditRepository.for_archive_root(tmp_path).ensure_archive_authority(now_ms=1)

    assert external_audit.read_bytes() == before


def test_audit_authority_rejects_a_foreign_owned_audit_leaf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The authority leaf must belong to this effective archive owner.

    Anti-vacuity: removing the uid comparison accepts the otherwise-valid
    single-linked regular file and allows the authority check to proceed.
    """

    bootstrap_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    before = audit_path.read_bytes()
    monkeypatch.setattr("polylogue.storage.sqlite.audit_leaf.os.geteuid", lambda: audit_path.stat().st_uid + 1)

    with pytest.raises(RuntimeError, match="current effective user"):
        AuditRepository.for_archive_root(tmp_path).ensure_archive_authority(now_ms=1)

    assert audit_path.read_bytes() == before


def test_audit_leaf_uses_the_verified_native_directory_when_descriptor_children_are_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A portable native descriptor path is used before pseudo-filesystem traversal.

    Anti-vacuity: removing the F_GETPATH-style route makes this macOS-shaped
    host fail closed because neither pseudo-filesystem child is available.
    """

    bootstrap_archive_root(tmp_path)

    def native_path_from_descriptor(_fd: int, _request: int, _buffer: bytes) -> bytes:
        return os.fsencode(tmp_path) + b"\0"

    monkeypatch.setattr(VerifiedAuditLeaf, "_descriptor_child_path", lambda _self: None)
    monkeypatch.setattr("polylogue.storage.sqlite.audit_leaf.fcntl.F_GETPATH", 50, raising=False)
    monkeypatch.setattr("polylogue.storage.sqlite.audit_leaf.fcntl.fcntl", native_path_from_descriptor)
    with VerifiedAuditLeaf(tmp_path) as leaf:
        assert leaf.anchored_path == tmp_path / "audit.db"


def test_audit_leaf_closes_its_directory_descriptor_after_validation_failure(tmp_path: Path) -> None:
    """Rejected leaves do not retain one descriptor per failed authority request.

    Anti-vacuity: the old OSError-only cleanup leaves ``_directory_fd`` set
    after this symlink validation error.
    """

    target = tmp_path.parent / "external-audit-leaf.db"
    target.write_bytes(b"external")
    (tmp_path / "audit.db").symlink_to(target)
    leaf = VerifiedAuditLeaf(tmp_path)

    with pytest.raises(AuditLeafError):
        leaf.__enter__()

    assert leaf._directory_fd is None
    assert leaf._leaf_fd is None


def test_audit_leaf_rejects_a_foreign_sidecar_without_leaking_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SQLite namespace is validated with the main leaf before any writer opens.

    Anti-vacuity: checking only audit.db accepts this foreign-owned WAL leaf
    and lets SQLite consume attacker-controlled sidecar bytes.
    """

    bootstrap_archive_root(tmp_path)
    sidecar = tmp_path / "audit.db-wal"
    sidecar.write_bytes(b"not a sqlite wal")
    leaf = VerifiedAuditLeaf(tmp_path)
    real_stat = sidecar.stat()
    real_os_stat = os.stat

    def foreign_sidecar_metadata(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes], *args: Any, **kwargs: Any
    ) -> os.stat_result:
        metadata = real_os_stat(path, *args, **kwargs)
        if path == "audit.db-wal":
            values = list(metadata)
            values[4] = real_stat.st_uid + 1
            return os.stat_result(values)
        return metadata

    monkeypatch.setattr("polylogue.storage.sqlite.audit_leaf.os.stat", foreign_sidecar_metadata)

    with pytest.raises(AuditLeafError, match="sidecar.*current effective user"):
        leaf.__enter__()

    assert leaf._directory_fd is None
    assert leaf._leaf_fd is None


def test_audit_leaf_rejects_group_writable_archive_directory(tmp_path: Path) -> None:
    """A second Unix principal cannot plant an SQLite sidecar in the authority namespace."""

    bootstrap_archive_root(tmp_path)
    tmp_path.chmod(0o770)
    leaf = VerifiedAuditLeaf(tmp_path)

    with pytest.raises(AuditLeafError, match="directory must not be writable by group or other"):
        leaf.__enter__()

    assert leaf._directory_fd is None
    assert leaf._leaf_fd is None


def test_audit_leaf_rejects_group_writable_main_and_sidecar_files(tmp_path: Path) -> None:
    """UID equality alone cannot grant exclusive write authority over SQLite files."""

    bootstrap_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    audit_path.chmod(0o660)
    with pytest.raises(AuditLeafError, match="audit tier must not be writable by group or other"):
        VerifiedAuditLeaf(tmp_path).__enter__()

    audit_path.chmod(0o600)
    sidecar = tmp_path / "audit.db-wal"
    sidecar.write_bytes(b"not a sqlite wal")
    sidecar.chmod(0o660)
    with pytest.raises(AuditLeafError, match="sidecar must not be writable by group or other"):
        VerifiedAuditLeaf(tmp_path).__enter__()


def test_audit_leaf_serializes_writers_across_the_main_and_sidecar_namespace(tmp_path: Path) -> None:
    """A second writer cannot validate then race the first SQLite namespace owner.

    Anti-vacuity: without the nonblocking directory lock, both contexts open
    and can independently create or replace the audit sidecar namespace.
    """

    bootstrap_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    with open_verified_audit_connection(audit_path):
        with pytest.raises(AuditLeafError, match="active writer"):
            with open_verified_audit_connection(audit_path):
                pass


def test_audit_authority_rejects_a_leaf_replaced_during_sqlite_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SQLite never yields a connection after the descriptor-checked leaf changes."""

    bootstrap_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    replacement = tmp_path / "replacement-audit.db"
    replacement.write_bytes(audit_path.read_bytes())
    before = replacement.read_bytes()
    original_connect = cast(Callable[..., sqlite3.Connection], sqlite3.connect)
    swapped = False

    def replace_after_open(database: object, *args: object, **kwargs: object) -> sqlite3.Connection:
        nonlocal swapped
        connection = original_connect(database, *args, **kwargs)
        database_text = str(database)
        if (
            not swapped
            and database_text.split("?", 1)[0].endswith("/audit.db")
            and ("/dev/fd/" in database_text or "/proc/self/fd/" in database_text)
        ):
            swapped = True
            audit_path.unlink()
            replacement.replace(audit_path)
        return connection

    monkeypatch.setattr("polylogue.storage.sqlite.audit_leaf.sqlite3.connect", replace_after_open)

    with pytest.raises(RuntimeError, match="changed during SQLite open"):
        AuditRepository.for_archive_root(tmp_path).ensure_archive_authority(now_ms=1)

    assert swapped
    assert audit_path.read_bytes() == before


def test_verified_audit_writer_rejects_a_wal_replacement_before_first_application_begin(tmp_path: Path) -> None:
    """The production writer pins WAL/SHM before a caller can start its transaction."""

    bootstrap_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    wal_path = audit_path.with_name("audit.db-wal")

    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        with open_verified_audit_connection(audit_path) as connection:
            replacement = tmp_path / "replacement-audit.db-wal"
            replacement.write_bytes(wal_path.read_bytes())
            replacement.replace(wal_path)
            connection.execute("BEGIN IMMEDIATE")


def test_verified_audit_reader_observes_a_committed_live_wal_head(tmp_path: Path) -> None:
    """Read-only authority checks include commits still resident in the live WAL."""

    bootstrap_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    archive_id = "archive:live-wal-read"

    with open_verified_audit_connection(audit_path) as writer:
        writer.execute(
            "INSERT INTO archive_authority(archive_instance_id, created_at_ms, authority_format) VALUES (?, 1, 1)",
            (archive_id,),
        )
        writer.commit()
        assert audit_path.with_name("audit.db-wal").exists()

        with open_verified_audit_read_connection(audit_path) as reader:
            assert reader.execute(
                "SELECT archive_instance_id FROM archive_authority WHERE archive_instance_id = ?", (archive_id,)
            ).fetchone() == (archive_id,)


@pytest.mark.parametrize(
    "statement",
    (
        "INSERT INTO archive_authority(archive_instance_id, created_at_ms, authority_format) VALUES ('new', 1, 1)",
        "UPDATE audit_continuity_head SET generation = generation + 1",
        "DELETE FROM archive_authority",
        "CREATE TABLE unauthorized_audit_table(value TEXT)",
        "PRAGMA user_version = 2",
        "ATTACH",
    ),
)
def test_verified_audit_reader_translates_sqlite_failure_without_writing(tmp_path: Path, statement: str) -> None:
    bootstrap_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    if statement == "ATTACH":
        statement = f"ATTACH DATABASE '{(tmp_path / 'auxiliary.db').as_uri()}?mode=rwc' AS auxiliary"
    with pytest.raises(AuditLeafError, match="audit SQLite read is unavailable") as failure:
        with open_verified_audit_read_connection(audit_path) as reader:
            assert reader.execute("PRAGMA query_only").fetchone() == (1,)
            reader.execute(statement)
    assert isinstance(failure.value.__cause__, sqlite3.DatabaseError)
    assert not (tmp_path / "auxiliary.db").exists()


def test_settled_audit_read_reports_sqlite_failure_as_pending(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.audit_continuity import AuditContinuityPendingError

    bootstrap_archive_root(tmp_path)
    audit = AuditRepository.for_archive_root(tmp_path)
    with pytest.raises(AuditContinuityPendingError, match="audit machine read is unavailable") as failure:
        with audit.settled_machine_read(), audit._connection() as reader:
            reader.execute("SELECT * FROM synthetic_missing_table")
    assert isinstance(failure.value.__cause__, AuditLeafError)
    assert isinstance(failure.value.__cause__.__cause__, sqlite3.OperationalError)


def test_verified_audit_writer_coexists_with_an_older_read_transaction(tmp_path: Path) -> None:
    """Persistent WAL mode lets a later writer proceed while a reader retains its snapshot."""

    bootstrap_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    with open_verified_audit_connection(audit_path) as writer:
        assert writer.execute("PRAGMA journal_mode").fetchone() == ("wal",)

    with open_verified_audit_read_connection(audit_path) as reader:
        reader.execute("BEGIN")
        reader.execute("SELECT generation FROM audit_continuity_head").fetchone()
        with open_verified_audit_connection(audit_path) as writer:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("UPDATE audit_continuity_head SET advanced_at_ms = advanced_at_ms")
            writer.commit()


def test_production_factory_does_not_abandon_a_live_same_process_attempt(tmp_path: Path) -> None:
    """A second composition-root call recognizes the first executor's owner."""
    bootstrap_archive_root(tmp_path)
    actuator = _Actuator()
    first = OperationExecutor.for_archive_root(tmp_path, token_factory=lambda: "first-owner-token")
    preview = first.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:live-owner",
        archive_identity_digest="identity:live-owner",
        parameter_digest="params:live-owner",
    )
    authorization = first.authorize_bound(_binding(actuator), preview, _principal())
    assert first._audit is not None
    operation_id = first._audit.consume_authorization_and_start(preview, authorization)

    OperationExecutor.for_archive_root(tmp_path)

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert (
            conn.execute(
                "SELECT state, worker_id FROM operation_attempts WHERE operation_id = ?", (operation_id,)
            ).fetchone()[0]
            == "running"
        )


def test_recovery_marks_a_dead_process_owned_attempt_unknown(tmp_path: Path) -> None:
    """Restart recovery remains active when the recorded owner no longer exists."""
    bootstrap_archive_root(tmp_path)
    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "dead-owner-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:dead-owner",
        archive_identity_digest="identity:dead-owner",
        parameter_digest="params:dead-owner",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    operation_id = audit.consume_authorization_and_start(preview, authorization)
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        conn.execute(
            "UPDATE operation_attempts SET worker_id = 'pid:999999999:0' WHERE operation_id = ?", (operation_id,)
        )
        conn.commit()

    assert audit.recover_abandoned_attempts() == (operation_id,)
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone() == (
            "interrupted",
        )


def _dead_nonterminal_operation(tmp_path: Path, actuator: _Actuator) -> tuple[AuditRepository, str]:
    """Create the production durable boundary immediately after attempt start."""

    audit = _audit(tmp_path)
    first = OperationExecutor(audit=audit, token_factory=lambda: "crash-boundary-token")
    binding = _binding(actuator)
    preview = first.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:recovery",
        archive_identity_digest="identity:recovery",
        parameter_digest="params:recovery",
    )
    authorization = first.authorize_bound(binding, preview, _principal())
    operation_id = audit.consume_authorization_and_start(preview, authorization)
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        conn.execute(
            "UPDATE operation_attempts SET worker_id = 'pid:999999999:0' WHERE operation_id = ?", (operation_id,)
        )
        conn.commit()
    return audit, operation_id


@dataclass
class _DestructiveClassActuator(_Actuator):
    """Like ``_Actuator``, but ``prepare()`` honors ``self.destructive_class``.

    ``_Actuator.prepare()`` hardcodes ``destructive_class="reversible"``
    regardless of the ``destructive_class`` field, so it cannot exercise the
    delete-class recovery-barrier bypass on its own.
    """

    def prepare(self, _args: object) -> MutationPlan:
        targets = ("session:changed",) if self.changed else self.target_refs
        return build_plan(
            operation=self.operation,
            destructive_class=self.destructive_class,
            target_refs=targets,
            affected_tiers=("user",),
            reversible=self.destructive_class in {"additive", "reversible"},
        )


def _register_fixture(monkeypatch: pytest.MonkeyPatch, actuator: _Actuator) -> None:
    from polylogue.operations import mutation_transaction

    monkeypatch.setitem(mutation_transaction._RECOVERY_ROUTES, actuator.operation, actuator)


def _retry(
    audit: AuditRepository, actuator: _Actuator, token: str, *, actor_ref: str = "actor:test"
) -> MutationReceipt:
    principal = replace(_principal(), actor_ref=actor_ref)
    retry = OperationExecutor(audit=audit, token_factory=lambda: token)
    binding = _binding(actuator)
    preview = retry.prepare_bound(
        binding,
        object(),
        principal,
        archive_instance_id="archive:recovery",
        archive_identity_digest="identity:recovery",
        parameter_digest="params:recovery",
    )
    authorization = retry.authorize_bound(binding, preview, principal)
    return retry.execute_bound(binding, preview, authorization, object())


def _run_state(tmp_path: Path, operation_id: str) -> tuple[str, str, tuple[str, ...]]:
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        status, reason = conn.execute(
            "SELECT status, terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        targets = tuple(
            str(row[0])
            for row in conn.execute(
                "SELECT state FROM operation_targets WHERE operation_id = ? ORDER BY ordinal", (operation_id,)
            )
        )
    return str(status), str(reason), targets


def test_dead_overlapping_attempt_is_replayed_before_the_new_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after attempt start is re-applied, then the new request proceeds.

    Anti-vacuity: skip ``_resolve_overlapping_operations`` and the dead run
    stays ``running``; stop terminalizing it and it is met again by the next
    request instead of reading ``recovered_complete``.
    """

    actuator = _Actuator()
    _register_fixture(monkeypatch, actuator)
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)

    receipt = _retry(audit, actuator, "retry-boundary-token", actor_ref="actor:resolver")

    assert receipt.status == "applied"
    assert (actuator.recoveries, actuator.calls) == (1, 2)
    assert _run_state(tmp_path, operation_id) == ("completed", "recovered_complete", ("applied",))
    assert audit.list_events(operation_id)[-1]["event_type"] == "recovery_resolved"
    assert audit.list_events(operation_id)[-1]["actor_ref"] == "actor:resolver"
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT actor_ref FROM operation_runs WHERE operation_id=?", (operation_id,)
        ).fetchone() == ("actor:test",)


def test_failed_replay_terminalizes_without_blocking_the_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replay that raises is a typed terminal failure, never a standing refusal.

    Anti-vacuity: let the replay exception escape ``resolve_interrupted_operation``
    and the new request raises instead of applying.
    """

    actuator = _Actuator(recovery_raises=True)
    _register_fixture(monkeypatch, actuator)
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)

    receipt = _retry(audit, actuator, "after-failed-replay-token")

    assert receipt.status == "applied"
    assert _run_state(tmp_path, operation_id) == ("failed", "recovery_replay_failed", ("failed",))


@pytest.mark.parametrize("registered", [False, True], ids=["unregistered-family", "retired-version"])
def test_unreplayable_interrupted_work_is_terminal_not_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered: bool
) -> None:
    """A family or version this runtime cannot replay ends failed, not ``unknown``.

    Anti-vacuity: drop the registry or version check in
    ``resolve_interrupted_operation`` and the retired plan is replayed through
    today's actuator (``recoveries`` becomes 1).
    """

    actuator = _Actuator()
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)
    retired = replace(actuator, operation_version=2)
    if registered:
        _register_fixture(monkeypatch, retired)

    recover_on_admitted_owner(tmp_path)

    assert (actuator.recoveries, retired.recoveries) == (0, 0)
    assert _run_state(tmp_path, operation_id) == ("failed", "recovery_not_replayable", ("failed",))


def test_startup_replays_an_interrupted_operation_to_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daemon startup resolves a dead operation on its own, with no request pending.

    Anti-vacuity: return early from ``recover_interrupted_operations`` after
    ``recover_abandoned_attempts`` and the run stays ``interrupted``.
    """

    actuator = _Actuator()
    _register_fixture(monkeypatch, actuator)
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)

    recover_on_admitted_owner(tmp_path)
    recover_on_admitted_owner(tmp_path)

    assert (actuator.recoveries, actuator.calls) == (1, 1)
    assert _run_state(tmp_path, operation_id) == ("completed", "recovered_complete", ("applied",))
    assert [event["event_type"] for event in audit.list_events(operation_id)].count("recovery_resolved") == 1
    assert audit.list_events(operation_id)[-1]["actor_ref"] == "daemon:recovery"


def test_recovery_resolution_replays_after_source_prepare_crash_at_daemon_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prepared resolution command survives a crash and is completed at startup.

    Anti-vacuity: drop the receipt from the write-ahead payload and the replayed
    resolution cannot be rebuilt, so this restart raises.
    """

    actuator = _Actuator()
    _register_fixture(monkeypatch, actuator)
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)
    original_phase = AuditContinuityCoordinator._phase

    def interrupt_resolution(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "record_recovery_resolution" and phase == "after_source_prepare":
            raise RuntimeError("crash after recovery resolution prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_resolution)
    with pytest.raises(RuntimeError, match="recovery resolution prepare"):
        _retry(audit, actuator, "crash-resolver-token", actor_ref="actor:crash-resolver")
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)

    recover_on_admitted_owner(tmp_path)

    assert _run_state(tmp_path, operation_id) == ("completed", "recovered_complete", ("applied",))
    assert audit.list_events(operation_id)[-1]["actor_ref"] == "actor:crash-resolver"
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone() == (None,)


def test_unknown_owner_is_not_stolen_by_an_overlapping_apply(tmp_path: Path) -> None:
    """An unreadable ownership witness is blocking, never an orphan shortcut."""

    actuator = _Actuator()
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        conn.execute(
            "UPDATE operation_attempts SET worker_id = 'external:unverifiable' WHERE operation_id = ?", (operation_id,)
        )
        conn.commit()
    retry = OperationExecutor(audit=audit, token_factory=lambda: "unknown-owner-token")
    binding = _binding(actuator)
    preview = retry.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:recovery",
        archive_identity_digest="identity:recovery",
        parameter_digest="params:recovery",
    )
    authorization = retry.authorize_bound(binding, preview, _principal())

    with pytest.raises(RecoveryBlockedError, match="unknown owner"):
        retry.execute_bound(binding, preview, authorization, object())

    assert actuator.recoveries == 0
    assert actuator.calls == 0
    assert audit.list_events(operation_id)[-1]["event_type"] == "authorization_consumed"


@pytest.mark.parametrize("owner_id", [None, "external:unverifiable"])
def test_recovery_preserves_attempts_with_unproven_owners(tmp_path: Path, owner_id: str | None) -> None:
    """Legacy or externally-owned attempts stay running until their owner is proven dead."""

    bootstrap_archive_root(tmp_path)
    audit = _audit(tmp_path)
    executor = OperationExecutor(audit=audit, token_factory=lambda: "unproven-owner-token")
    preview = executor.prepare_bound(
        _binding(_Actuator()),
        object(),
        _principal(),
        archive_instance_id="archive:unproven-owner",
        archive_identity_digest="identity:unproven-owner",
        parameter_digest="params:unproven-owner",
    )
    authorization = executor.authorize_bound(_binding(_Actuator()), preview, _principal())
    operation_id = audit.consume_authorization_and_start(preview, authorization)
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        conn.execute("UPDATE operation_attempts SET worker_id = ? WHERE operation_id = ?", (owner_id, operation_id))
        conn.commit()

    assert audit.recover_abandoned_attempts() == ()
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone() == (
            "running",
        )


def test_process_owner_liveness_is_unknown_when_start_ticks_cannot_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live PID with unreadable identity evidence is not proof that its owner died."""

    monkeypatch.setattr("polylogue.operations.audit.os.kill", lambda _pid, _signal: None)
    monkeypatch.setattr(Path, "read_text", lambda _self, *, encoding: (_ for _ in ()).throw(OSError("denied")))

    assert _attempt_owner_liveness("pid:321:known-start") == "unknown"


def test_process_owner_uses_proc_start_ticks_after_a_spaced_process_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """PID reuse remains detectable when /proc's parenthesized comm has spaces."""

    state = {"stat": "321 (worker process) S " + " ".join(["0"] * 17 + ["stable", "old", "0"])}

    def read_text(self: Path, *, encoding: str) -> str:
        assert self == Path("/proc/321/stat")
        assert encoding == "utf-8"
        return state["stat"]

    monkeypatch.setattr("polylogue.operations.audit.os.getpid", lambda: 321)
    monkeypatch.setattr("polylogue.operations.audit.os.kill", lambda _pid, _signal: None)
    monkeypatch.setattr(Path, "read_text", read_text)

    owner = _current_process_attempt_owner()
    assert owner == "pid:321:old"

    state["stat"] = "321 (worker process) S " + " ".join(["0"] * 17 + ["stable", "new", "0"])
    assert not _attempt_owner_is_live(owner)


def test_audit_repository_cannot_bypass_the_continuity_coordinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(tmp_path)

    def reject_bypass(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("coordinator required")

    monkeypatch.setattr(AuditContinuityCoordinator, "execute", reject_bypass)
    with pytest.raises(RuntimeError, match="coordinator required"):
        audit.ensure_archive_authority(now_ms=1)


def test_audit_repository_replays_a_prepared_mutation_with_its_original_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(tmp_path)
    original_phase = AuditContinuityCoordinator._phase

    def interrupt_after_prepare(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if phase == "after_source_prepare":
            raise RuntimeError("crash after prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_after_prepare)
    with pytest.raises(RuntimeError, match="crash after prepare"):
        audit.ensure_archive_authority(now_ms=123, archive_instance_id="archive:replayed")
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)

    AuditRepository.for_archive_root(tmp_path).reconcile_continuity()
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT archive_instance_id, created_at_ms FROM archive_authority").fetchone() == (
            "archive:replayed",
            123,
        )


def test_reconcile_waits_for_an_inflight_continuity_write(tmp_path: Path) -> None:
    """Recovery cannot promote another writer's prepared command out from under it."""

    _audit(tmp_path)
    prepared = threading.Event()
    release_writer = threading.Event()
    reconcile_started = threading.Event()
    applied_by: list[str] = []

    def pause_after_prepare(phase: str, _mutation: AuditMutation) -> None:
        if phase == "after_source_prepare":
            prepared.set()
            if not release_writer.wait(timeout=10):
                raise TimeoutError("writer was not released")

    writer = AuditContinuityCoordinator(tmp_path, phase_hook=pause_after_prepare)
    recovery = AuditContinuityCoordinator(tmp_path)
    mutation = AuditMutation("test_serialized_recovery", "mutation:writer", 1, {})

    def execute_writer() -> None:
        writer.execute(mutation, lambda _connection, _mutation: applied_by.append("writer"))

    def reconcile() -> None:
        reconcile_started.set()
        recovery.reconcile(lambda _connection, _mutation: applied_by.append("recovery"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer_result = pool.submit(execute_writer)
        assert prepared.wait(timeout=10)
        reconcile_result = pool.submit(reconcile)
        assert reconcile_started.wait(timeout=10)
        with pytest.raises(FuturesTimeoutError):
            reconcile_result.result(timeout=0.1)
        release_writer.set()
        writer_result.result(timeout=10)
        reconcile_result.result(timeout=10)

    assert applied_by == ["writer"]
    with sqlite3.connect(tmp_path / "source.db") as source, sqlite3.connect(tmp_path / "audit.db") as audit:
        source_head = source.execute(
            "SELECT committed_generation, committed_head_sha256, pending_mutation_id "
            "FROM audit_continuity_control WHERE singleton = 1"
        ).fetchone()
        audit_head = audit.execute(
            "SELECT generation, head_sha256, mutation_id FROM audit_continuity_head WHERE singleton = 1"
        ).fetchone()
    assert source_head == (audit_head[0], audit_head[1], None)
    assert audit_head[2] == mutation.mutation_id


def test_mark_preview_stale_advances_and_replays_durable_continuity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "stale-continuity-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:stale-continuity",
        archive_identity_digest="identity:stale-continuity",
        parameter_digest="params:stale-continuity",
    )
    executor.authorize_bound(_binding(actuator), preview, _principal())
    with sqlite3.connect(tmp_path / "source.db") as source:
        before_generation = int(
            source.execute("SELECT committed_generation FROM audit_continuity_control").fetchone()[0]
        )

    original_phase = AuditContinuityCoordinator._phase

    def interrupt_after_prepare(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "mark_preview_stale" and phase == "after_source_prepare":
            raise RuntimeError("crash after stale-preview prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_after_prepare)
    with pytest.raises(RuntimeError, match="stale-preview prepare"):
        audit.mark_preview_stale(preview)
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)

    AuditRepository.for_archive_root(tmp_path).reconcile_continuity()

    with sqlite3.connect(tmp_path / "source.db") as source, sqlite3.connect(tmp_path / "audit.db") as audit_db:
        source_head = source.execute(
            "SELECT committed_generation, committed_head_sha256, pending_mutation_id FROM audit_continuity_control"
        ).fetchone()
        audit_head = audit_db.execute("SELECT generation, head_sha256 FROM audit_continuity_head").fetchone()
        assert source_head == (before_generation + 1, audit_head[1], None)
        assert audit_head[0] == before_generation + 1
        assert audit_db.execute(
            "SELECT state FROM operation_previews WHERE preview_id = ?", (preview.preview_ref,)
        ).fetchone() == ("stale",)
        assert audit_db.execute(
            "SELECT state FROM operation_authorizations WHERE preview_id = ?", (preview.preview_ref,)
        ).fetchone() == ("revoked",)


def test_replayed_start_keeps_the_crashed_owner_recoverable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recovery never adopts an actuator-less pre-effect attempt into its own process."""

    bootstrap_archive_root(tmp_path)
    crashed_owner = "pid:999999999:0"
    first = AuditRepository.for_archive_root(tmp_path, attempt_owner_id=crashed_owner)
    executor = OperationExecutor(audit=first, token_factory=lambda: "replayed-owner-token")
    preview = executor.prepare_bound(
        _binding(_Actuator()),
        object(),
        _principal(),
        archive_instance_id="archive:replayed-owner",
        archive_identity_digest="identity:replayed-owner",
        parameter_digest="params:replayed-owner",
    )
    authorization = executor.authorize_bound(_binding(_Actuator()), preview, _principal())
    original_phase = AuditContinuityCoordinator._phase

    def interrupt_start(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "consume_authorization_and_start" and phase == "after_source_prepare":
            raise RuntimeError("crash after start prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_start)
    with pytest.raises(RuntimeError, match="crash after start prepare"):
        first.consume_authorization_and_start(preview, authorization)
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)

    recovery = AuditRepository.for_archive_root(tmp_path, attempt_owner_id="pid:12345:recovery")
    recovery.reconcile_continuity()
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        operation_id = str(conn.execute("SELECT operation_id FROM operation_runs").fetchone()[0])
        assert conn.execute(
            "SELECT worker_id FROM operation_attempts WHERE operation_id = ?", (operation_id,)
        ).fetchone() == (crashed_owner,)

    assert recovery.recover_abandoned_attempts() == (operation_id,)
    assert recovery.get_operation(operation_id)["status"] == "interrupted"  # type: ignore[index]


def test_optional_archive_authority_id_replays_without_changing_existing_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An omitted authority id remains omitted across a pre-commit crash."""

    audit = _audit(tmp_path)
    assert audit.ensure_archive_authority(now_ms=1, archive_instance_id="archive:existing") == "archive:existing"
    original_phase = AuditContinuityCoordinator._phase

    def interrupt_after_prepare(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "ensure_archive_authority" and phase == "after_source_prepare":
            raise RuntimeError("crash after optional authority prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_after_prepare)
    with pytest.raises(RuntimeError, match="optional authority prepare"):
        audit.ensure_archive_authority(now_ms=2)
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)

    AuditRepository.for_archive_root(tmp_path).reconcile_continuity()
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT archive_instance_id, created_at_ms FROM archive_authority").fetchone() == (
            "archive:existing",
            1,
        )


def test_typed_domain_receipt_replays_after_source_prepare_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real executor route persists and replays typed receipt values as JSON."""

    audit = _audit(tmp_path)
    actuator = _TypedReceiptActuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "typed-receipt-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:typed-receipt",
        archive_identity_digest="identity:typed-receipt",
        parameter_digest="params:typed-receipt",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    original_phase = AuditContinuityCoordinator._phase

    def interrupt_finalize(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "finalize_attempt" and phase == "after_source_prepare":
            raise RuntimeError("crash after typed receipt prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_finalize)
    with pytest.raises(AuditFinalizationError, match="not reported completed"):
        executor.execute_bound(_binding(actuator), preview, authorization, object())
    with sqlite3.connect(tmp_path / "source.db") as source:
        pending_payload = str(source.execute("SELECT pending_payload_json FROM audit_continuity_control").fetchone()[0])
    assert "private-cache" not in pending_payload
    assert "annotation-batch:typed" not in pending_payload
    assert '"domain_receipt"' not in pending_payload
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)

    AuditRepository.for_archive_root(tmp_path).reconcile_continuity()
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT status FROM operation_runs").fetchone() == ("completed",)
        receipt_json = str(
            conn.execute("SELECT detail_json FROM operation_events WHERE event_type = 'attempt_finalized'").fetchone()[
                0
            ]
        )
    assert "private-cache" not in receipt_json
    detail = json.loads(receipt_json)
    assert detail["affected_count"] == 1
    assert "domain_receipt" not in detail
    assert "annotation-batch:typed" not in receipt_json
    with sqlite3.connect(tmp_path / "source.db") as source:
        command = source.execute("SELECT pending_payload_json FROM audit_continuity_control").fetchone()[0]
    assert command is None


def test_closed_historical_receipt_replays_through_source_wal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A terminal machine fact survives the only crash window before audit commit.

    Anti-vacuity: removing ``historical_receipt`` from the continuity payload
    makes the reconciled final event lack the receipt this accessor returns.
    """

    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "historical-receipt-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:historical-receipt",
        archive_identity_digest="identity:historical-receipt",
        parameter_digest="params:historical-receipt",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    started = executor.begin_bound(_binding(actuator), preview, authorization, object())
    history = InsightPartHistoricalReceipt(
        ordinal=0,
        page_count=1,
        manifest_digest="a" * 64,
        index_generation="index-generation:fixture",
        recipe_version="fixture-recipe",
        targets=[
            InsightTargetHistoricalReceipt(
                target_ref="session:fixture",
                disposition="published",
                input_binding="input:fixture",
                output_binding="output:fixture",
                certified_counts=InsightCertifiedCountsHistorical(profiles=1),
                publication_known_committed=True,
            )
        ],
    )
    receipt = MutationReceipt(
        operation=started.plan.operation,
        plan_hash=started.plan.plan_hash,
        status="applied",
        target_refs=started.plan.target_refs,
        affected_count=1,
        detail=None,
        receipt_ref=None,
        applied_at="now",
        historical_receipt=history,
    )
    original_phase = AuditContinuityCoordinator._phase

    def interrupt_finalize(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "finalize_attempt" and phase == "after_source_prepare":
            raise RuntimeError("crash after historical receipt prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_finalize)
    with pytest.raises(AuditFinalizationError, match="not reported completed"):
        executor.finalize_bound(started, receipt=receipt)
    with sqlite3.connect(tmp_path / "source.db") as source:
        pending = str(source.execute("SELECT pending_payload_json FROM audit_continuity_control").fetchone()[0])
    assert '"kind":"insight-part/v1"' in pending
    assert '"domain_receipt"' not in pending
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)

    AuditRepository.for_archive_root(tmp_path).reconcile_continuity()
    replayed = AuditRepository.for_archive_root(tmp_path).historical_machine_receipt(str(started.operation_id))
    assert replayed == history


def test_atomic_batch_finalization_marks_every_target_and_terminates_run(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    actuator = _Actuator(target_refs=("session:first", "session:second"))
    executor = OperationExecutor(audit=audit, token_factory=lambda: "batch-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:batch",
        archive_identity_digest="identity:batch",
        parameter_digest="params:batch",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())

    receipt = executor.execute_bound(_binding(actuator), preview, authorization, object())

    assert receipt.operation_id is not None
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT status, affected_count, unknown_count FROM operation_runs WHERE operation_id = ?",
            (receipt.operation_id,),
        ).fetchone() == ("completed", 2, 0)
        assert conn.execute(
            "SELECT state FROM operation_targets WHERE operation_id = ? ORDER BY ordinal",
            (receipt.operation_id,),
        ).fetchall() == [("applied",), ("applied",)]
        assert conn.execute(
            "SELECT domain_receipt_ref, domain_receipt_kind FROM operation_runs WHERE operation_id = ?",
            (receipt.operation_id,),
        ).fetchone() == (receipt.receipt_ref, "mutation-receipt")
        assert conn.execute(
            "SELECT domain_receipt_ref, domain_receipt_kind FROM operation_targets WHERE operation_id = ? ORDER BY ordinal",
            (receipt.operation_id,),
        ).fetchall() == [(receipt.receipt_ref, "mutation-receipt"), (receipt.receipt_ref, "mutation-receipt")]


def test_finalization_rejects_a_receipt_not_bound_to_the_durable_plan(tmp_path: Path) -> None:
    """A domain receipt cannot finalize a different operation or target set."""

    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "receipt-binding-token")
    binding = _binding(actuator)
    preview = executor.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:receipt-binding",
        archive_identity_digest="identity:receipt-binding",
        parameter_digest="params:receipt-binding",
    )
    authorization = executor.authorize_bound(binding, preview, _principal())
    operation_id = audit.consume_authorization_and_start(preview, authorization)
    invalid = MutationReceipt(
        operation=preview.plan.operation,
        plan_hash="wrong-plan",
        status="applied",
        target_refs=preview.plan.target_refs,
        affected_count=1,
        detail=None,
        receipt_ref="domain:wrong-plan",
        applied_at="now",
    )

    with pytest.raises(ValueError, match="plan"):
        audit.finalize_attempt(operation_id, status="applied", receipt=invalid)

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone() == (
            "running",
        )


def test_zero_target_finalization_completes_a_successful_noop(tmp_path: Path) -> None:
    """The real start/finalize route terminalizes a successful empty target set."""

    audit = _audit(tmp_path)
    actuator = _Actuator(target_refs=())
    executor = OperationExecutor(audit=audit, token_factory=lambda: "zero-target-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:zero-target",
        archive_identity_digest="identity:zero-target",
        parameter_digest="params:zero-target",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())

    receipt = executor.execute_bound(_binding(actuator), preview, authorization, object())

    assert receipt.operation_id is not None
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT status, terminal_reason, affected_count FROM operation_runs WHERE operation_id = ?",
            (receipt.operation_id,),
        ).fetchone() == ("completed", None, 0)


def test_expired_authorization_is_durably_marked_before_execute_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound execution route commits expiry instead of rolling it back with the refusal."""

    audit = _audit(tmp_path)
    clock = [1_000]
    executor = OperationExecutor(audit=audit, now_ms=lambda: clock[0], token_factory=lambda: "expired-token")
    preview = executor.prepare_bound(
        _binding(_Actuator()),
        object(),
        _principal(),
        archive_instance_id="archive:expired",
        archive_identity_digest="identity:expired",
        parameter_digest="params:expired",
        expires_at_ms=61_000,
    )
    authorization = executor.authorize_bound(_binding(_Actuator()), preview, _principal())
    clock[0] = 61_000
    monkeypatch.setattr("polylogue.operations.audit.time.time", lambda: 61.0)

    with pytest.raises(TokenExpiredError):
        executor.execute_bound(_binding(_Actuator()), preview, authorization, object())

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT state FROM operation_authorizations").fetchone() == ("expired",)


def test_authorization_expiry_is_canonicalized_from_the_durable_preview(tmp_path: Path) -> None:
    """A caller cannot issue a longer-lived bearer than the preview authorizes.

    Anti-vacuity: storing ``authorization.expires_at_ms`` accepts the forged
    expiry and leaves an authorization row that outlives its durable preview.
    """

    audit = _audit(tmp_path)
    clock = [1_000]
    executor = OperationExecutor(audit=audit, now_ms=lambda: clock[0], token_factory=lambda: "canonical-expiry")
    preview = executor.prepare_bound(
        _binding(_Actuator()),
        object(),
        _principal(),
        archive_instance_id="archive:canonical-expiry",
        archive_identity_digest="identity:canonical-expiry",
        parameter_digest="params:canonical-expiry",
        expires_at_ms=2_000,
    )
    authorization = executor.authorize_bound(_binding(_Actuator()), preview, _principal())

    with pytest.raises(ValueError, match="evidence differs"):
        audit.issue_authorization(
            preview,
            _principal(),
            replace(authorization, token="forged-expiry", expires_at_ms=3_000),
            issued_at_ms=1_100,
        )

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT expires_at_ms FROM operation_previews").fetchone() == (2_000,)
        assert conn.execute("SELECT expires_at_ms FROM operation_authorizations").fetchone() == (2_000,)


def test_authorization_consumption_uses_durable_actor_and_capability_evidence(tmp_path: Path) -> None:
    """Execution refuses reconstructed authority that differs from durable rows.

    Anti-vacuity: persisting run fields from the caller object records this
    substituted actor/capability instead of the issued authorization evidence.
    """

    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "durable-evidence")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:durable-evidence",
        archive_identity_digest="identity:durable-evidence",
        parameter_digest="params:durable-evidence",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    forged = replace(
        authorization,
        actor="actor:substituted",
        role="administrator",
        capability="archive.substituted.write",
        capabilities=("archive.substituted.write",),
    )

    with pytest.raises(AuthorizationMismatchError, match="principal mismatch"):
        audit.consume_authorization_and_start(preview, forged)

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM operation_runs").fetchone() == (0,)
        assert conn.execute("SELECT state FROM operation_authorizations").fetchone() == ("active",)
        assert conn.execute("SELECT actor_ref FROM operation_authorizations").fetchone() == ("actor:test",)
        assert conn.execute("SELECT capability FROM operation_authorization_capabilities").fetchone() == (
            "archive.fixture.write",
        )


def test_authorization_replay_preserves_the_prepared_expiry_and_issue_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retried WAL authorization reuses its prepared durable evidence verbatim.

    Anti-vacuity: omitting ``issued_at_ms`` during replay silently substitutes
    the recovery clock and can make an authorization valid longer than the
    original prepared command proved.
    """

    audit = _audit(tmp_path)
    clock = [1_000]
    executor = OperationExecutor(audit=audit, now_ms=lambda: clock[0], token_factory=lambda: "replayed-expiry")
    preview = executor.prepare_bound(
        _binding(_Actuator()),
        object(),
        _principal(),
        archive_instance_id="archive:replayed-expiry",
        archive_identity_digest="identity:replayed-expiry",
        parameter_digest="params:replayed-expiry",
        expires_at_ms=2_000,
    )
    original_abort = AuditContinuityCoordinator._abort_prepared

    def interrupt_issue(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "issue_authorization" and phase == "after_source_prepare":
            raise RuntimeError("crash after authorization prepare")

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_issue)
    monkeypatch.setattr(AuditContinuityCoordinator, "_abort_prepared", lambda _self, _prepared: None)
    with pytest.raises(RuntimeError, match="authorization prepare"):
        executor.authorize_bound(_binding(_Actuator()), preview, _principal())

    monkeypatch.setattr(AuditContinuityCoordinator, "_abort_prepared", original_abort)
    audit.reconcile_continuity()

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT issued_at_ms, expires_at_ms FROM operation_authorizations").fetchone() == (
            1_000,
            2_000,
        )


def test_already_satisfied_receipt_preserves_target_state_and_zero_affected_count(tmp_path: Path) -> None:
    """A nonempty idempotent success remains distinct from an applied domain effect."""

    audit = _audit(tmp_path)
    actuator = _AlreadySatisfiedActuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "already-satisfied-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:already-satisfied",
        archive_identity_digest="identity:already-satisfied",
        parameter_digest="params:already-satisfied",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    receipt = executor.execute_bound(_binding(actuator), preview, authorization, object())

    assert receipt.operation_id is not None
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT state FROM operation_targets WHERE operation_id = ?", (receipt.operation_id,)
        ).fetchone() == ("already_satisfied",)
        assert conn.execute(
            "SELECT status, affected_count FROM operation_runs WHERE operation_id = ?", (receipt.operation_id,)
        ).fetchone() == (
            "completed",
            0,
        )


def test_failed_receipt_marks_the_audit_attempt_failed(tmp_path: Path) -> None:
    """A domain-declared failure cannot leave an applied attempt receipt.

    Anti-vacuity: restoring the rejected-only attempt mapping makes the target
    failed while this attempt row incorrectly returns applied.
    """

    audit = _audit(tmp_path)
    actuator = _FailedReceiptActuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "failed-receipt-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:failed-receipt",
        archive_identity_digest="identity:failed-receipt",
        parameter_digest="params:failed-receipt",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    receipt = executor.execute_bound(_binding(actuator), preview, authorization, object())

    assert receipt.operation_id is not None
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT state FROM operation_attempts WHERE operation_id = ?", (receipt.operation_id,)
        ).fetchone() == ("failed",)
        assert conn.execute(
            "SELECT status FROM operation_runs WHERE operation_id = ?", (receipt.operation_id,)
        ).fetchone() == ("failed",)


def test_tampered_preview_payload_refuses_before_audit_intent(tmp_path: Path) -> None:
    """Execution cannot journal targets substituted into a reconstructed preview.

    Anti-vacuity: deleting the typed hash validation consumes the token and
    creates an audit run whose target set comes from the altered preview.
    """

    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "tampered-preview-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:tampered-preview",
        archive_identity_digest="identity:tampered-preview",
        parameter_digest="params:tampered-preview",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    tampered_preview = replace(preview, plan=replace(preview.plan, targets=()))

    with pytest.raises(AuthorizationMismatchError, match="authority hash"):
        executor.execute_bound(_binding(actuator), tampered_preview, authorization, object())

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM operation_runs").fetchone() == (0,)
        assert conn.execute("SELECT state FROM operation_authorizations").fetchone() == ("active",)


def test_blocked_finalization_rejects_targets_and_fails_parent_run(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "blocked-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:blocked",
        archive_identity_digest="identity:blocked",
        parameter_digest="params:blocked",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    operation_id = audit.consume_authorization_and_start(preview, authorization)

    audit.finalize_attempt(operation_id, status="blocked")

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT status, terminal_reason, rejected_count FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("failed", "target_rejected", 1)
        assert conn.execute(
            "SELECT state FROM operation_targets WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("rejected",)


def test_invalid_capability_and_stale_preview_refuse_before_apply(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )
    with pytest.raises(CapabilityDeniedError):
        executor.authorize_bound(
            _binding(actuator),
            preview,
            MutationPrincipal("actor:bad", frozenset(), "internal"),
        )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    actuator.changed = True
    with pytest.raises(PlanStaleError):
        executor.execute_bound(_binding(actuator), preview, authorization, object())
    assert actuator.calls == 0
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT state FROM operation_previews WHERE preview_id = ?", (preview.preview_ref,)
        ).fetchone() == ("stale",)
        assert conn.execute(
            "SELECT state FROM operation_authorizations WHERE authorization_id = ?",
            (authorization.authorization_id,),
        ).fetchone() == ("revoked",)
    actuator.changed = False
    with pytest.raises(TokenConsumedError):
        executor.execute_bound(_binding(actuator), preview, authorization, object())


def test_new_authorization_revokes_every_older_token_for_preview(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    tokens = iter(("first-token", "second-token"))
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: next(tokens))
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )

    first = executor.authorize_bound(_binding(actuator), preview, _principal())
    second = executor.authorize_bound(_binding(actuator), preview, _principal())

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        states = conn.execute(
            "SELECT authorization_id, state FROM operation_authorizations WHERE preview_id = ? ORDER BY issued_at_ms, authorization_id",
            (preview.preview_ref,),
        ).fetchall()
    assert {state for _authorization_id, state in states} == {"active", "revoked"}
    assert sum(state == "active" for _authorization_id, state in states) == 1
    with pytest.raises(TokenConsumedError):
        executor.execute_bound(_binding(actuator), preview, first, object())
    receipt = executor.execute_bound(_binding(actuator), preview, second, object())
    assert receipt.status == "applied"


def test_crash_after_intent_is_queryable_unknown_and_never_completed(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    actuator = _Actuator(crash=True)
    executor = OperationExecutor(audit=audit, token_factory=lambda: "crash-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    with pytest.raises(RuntimeError, match="simulated"):
        executor.execute_bound(_binding(actuator), preview, authorization, object())
    conn = sqlite3.connect(tmp_path / "audit.db")
    try:
        operation_id = str(conn.execute("SELECT operation_id FROM operation_runs").fetchone()[0])
        status = str(
            conn.execute("SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone()[0]
        )
        target_state = str(
            conn.execute("SELECT state FROM operation_targets WHERE operation_id = ?", (operation_id,)).fetchone()[0]
        )
    finally:
        conn.close()
    assert status == "interrupted"
    assert target_state == "unknown"
    assert audit.list_events(operation_id)[-1]["event_type"] == "attempt_unknown"


def test_token_consumption_and_initial_attempt_roll_back_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "rollback-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())

    def fail_event(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected audit transaction failure")

    monkeypatch.setattr(AuditRepository, "_append_event", staticmethod(fail_event))
    with pytest.raises(RuntimeError, match="injected"):
        audit.consume_authorization_and_start(preview, authorization)

    conn = sqlite3.connect(tmp_path / "audit.db")
    try:
        auth_state = str(
            conn.execute(
                "SELECT state FROM operation_authorizations WHERE preview_id = ?", (preview.preview_ref,)
            ).fetchone()[0]
        )
        run_count = int(conn.execute("SELECT COUNT(*) FROM operation_runs").fetchone()[0])
        preview_state = str(
            conn.execute(
                "SELECT state FROM operation_previews WHERE preview_id = ?", (preview.preview_ref,)
            ).fetchone()[0]
        )
    finally:
        conn.close()
    assert auth_state == "active"
    assert preview_state == "prepared"
    assert run_count == 0


def _excise_binding(actuator: _Actuator) -> OperationBinding[object, object]:
    """Like ``_delete_binding``, but the effective destructive class is ``excise``."""

    spec = OperationSpec(
        name="mutate-fixture",
        kind=OperationKind.MAINTENANCE,
        description="fixture",
        mutates_state=True,
        executor_status="executor-routed",
        allowed_surfaces=("internal",),
        target_authority=(
            TargetAuthorityPolicy(
                key="session",
                target_kinds=("session",),
                required_capabilities=("archive.fixture.write",),
                destructive_class="excise",
                required_confirmation="bound_token",
                allowed_durabilities=("derived",),
                allowed_recovery=("none",),
            ),
        ),
        affected_tiers=("user",),
    )
    return OperationBinding(spec, actuator)


def test_machine_request_dedup_survives_an_index_generation_promotion(tmp_path: Path) -> None:
    """polylogue-cois9 (B): a replayed request id stays deduped across a promotion.

    ``machine_requests`` is durable audit state whose primary key leads with
    the live archive identity digest.  After a daemon restart across an index
    generation promotion the binding is re-derived from the new live identity,
    so a primary-key lookup silently returned ``None`` and the retried request
    re-executed as new.  The audit database is itself the durable archive
    scope, so ``request_id`` is the durable dedup key within it, and the row's
    stored identity is returned so the caller can rebind to the key its
    ``machine_request_parts`` already use.

    Anti-vacuity: restoring ``WHERE archive_identity = ? AND request_id = ?``
    makes this test red -- ``promoted`` becomes ``None`` and the conflicting
    principal/fingerprint lookups stop raising.
    """

    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "machine-promotion-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:fixture",
        archive_identity_digest="identity:generation-one",
        parameter_digest="params:fixture",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    binding = MachineRequestBinding(
        "identity:generation-one", "request:promotion", "actor:test", "a" * 64, "mutation.fixture"
    )
    with audit.bind_machine_request(binding, transition="consume_authorization_and_start"):
        executor.execute_bound(_binding(actuator), preview, authorization, object())

    restarted = AuditRepository.for_archive_root(tmp_path)
    promoted = restarted.machine_request(replace(binding, archive_identity="identity:generation-two"))
    assert promoted is not None
    assert promoted["request_id"] == "request:promotion"
    # The stored key is the one the durable rows (and their parts) use.
    assert promoted["archive_identity"] == "identity:generation-one"
    assert (
        restarted.machine_request_for_principal("identity:generation-two", "request:promotion", "actor:test")
        is not None
    )
    # A moved generation is not conflicting intent, but a different principal
    # or fingerprint still is.
    with pytest.raises(MachineRequestConflictError):
        restarted.machine_request(replace(binding, archive_identity="identity:generation-two", fingerprint="b" * 64))
    with pytest.raises(MachineRequestConflictError):
        restarted.machine_request(
            replace(binding, archive_identity="identity:generation-two", principal_ref="actor:other")
        )
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM machine_requests").fetchone()[0] == 1


@dataclass
class _IngestPageActuator(_Actuator):
    operation: str = INGEST_OPERATION

    def prepare(self, _args: object) -> MutationPlan:
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=self.target_refs,
            affected_tiers=("user",),
            reversible=True,
            context={
                "source_generation_id": "source-generation:fixture",
                "manifest_digest": "a" * 64,
                "enumeration_fingerprint": "b" * 64,
                "input_count": 1,
                "recipe_version": "fixture-recipe",
            },
        )


def test_ingest_session_id_pages_survive_restart_and_validate_identity(tmp_path: Path) -> None:
    """A terminal page reference resolves every changed ID from audit history."""

    audit = _audit(tmp_path)
    actuator = _IngestPageActuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "ingest-page-token")
    preview = executor.prepare_bound(
        _binding(actuator, operation_name=INGEST_OPERATION),
        object(),
        _principal(),
        archive_instance_id="archive:ingest-pages",
        archive_identity_digest="identity:ingest-pages",
        parameter_digest="params:ingest-pages",
    )
    authorization = executor.authorize_bound(_binding(actuator, operation_name=INGEST_OPERATION), preview, _principal())
    started = executor.begin_bound(
        _binding(actuator, operation_name=INGEST_OPERATION), preview, authorization, object()
    )
    assert started.operation_id is not None
    session_ids = [f"chatgpt:{index:05d}" for index in range(257)]
    audit.append_ingest_session_id_page(started.operation_id, 0, tuple(session_ids[:256]))
    audit.append_ingest_session_id_page(started.operation_id, 1, tuple(session_ids[256:]))

    restarted = AuditRepository.for_archive_root(tmp_path)
    assert (
        restarted.read_ingest_session_id_pages(
            started.operation_id,
            page_count=2,
            session_count=257,
            digest=ingest_session_ids_digest(session_ids),
        )
        == session_ids
    )
    with pytest.raises(ValueError, match="differ from terminal receipt"):
        restarted.read_ingest_session_id_pages(
            started.operation_id,
            page_count=2,
            session_count=257,
            digest="a" * 64,
        )
    with pytest.raises(ValueError, match="contiguous"):
        audit.append_ingest_session_id_page(started.operation_id, 3, ("chatgpt:00258",))


def test_ingest_session_id_page_replays_after_source_prepare_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(tmp_path)
    actuator = _IngestPageActuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "replayed-ingest-page-token")
    preview = executor.prepare_bound(
        _binding(actuator, operation_name=INGEST_OPERATION),
        object(),
        _principal(),
        archive_instance_id="archive:replayed-ingest-page",
        archive_identity_digest="identity:replayed-ingest-page",
        parameter_digest="params:replayed-ingest-page",
    )
    authorization = executor.authorize_bound(_binding(actuator, operation_name=INGEST_OPERATION), preview, _principal())
    started = executor.begin_bound(
        _binding(actuator, operation_name=INGEST_OPERATION), preview, authorization, object()
    )
    assert started.operation_id is not None
    original_phase = AuditContinuityCoordinator._phase

    def interrupt_page(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "append_ingest_session_id_page" and phase == "after_source_prepare":
            raise RuntimeError("crash after ingest ID page prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_page)
    with pytest.raises(RuntimeError, match="crash after ingest ID page prepare"):
        audit.append_ingest_session_id_page(started.operation_id, 0, ("chatgpt:00000",))
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)
    restarted = AuditRepository.for_archive_root(tmp_path)
    restarted.reconcile_continuity()
    assert restarted.read_ingest_session_id_pages(
        started.operation_id,
        page_count=1,
        session_count=1,
        digest=ingest_session_ids_digest(["chatgpt:00000"]),
    ) == ["chatgpt:00000"]
    restarted.append_ingest_session_id_page(started.operation_id, 0, ("chatgpt:00000",))
    assert (
        len(
            [
                event
                for event in restarted.list_events(started.operation_id)
                if event["event_type"] == "ingest_session_id_page"
            ]
        )
        == 1
    )


@pytest.mark.asyncio
async def test_paged_ingest_projection_restores_full_public_parse_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four changed IDs cross a three-ID inline threshold without losing the fourth."""

    from polylogue.api import Polylogue
    from polylogue.operations import machine_receipts

    monkeypatch.setattr(machine_receipts, "MAX_INLINE_INGEST_SESSION_IDS", 3)

    def prepare() -> tuple[str, list[str]]:
        audit = _audit(tmp_path)
        actuator = _IngestPageActuator()
        executor = OperationExecutor(audit=audit, token_factory=lambda: "api-ingest-page-token")
        preview = executor.prepare_bound(
            _binding(actuator, operation_name=INGEST_OPERATION),
            object(),
            _principal(),
            archive_instance_id="archive:api-ingest-pages",
            archive_identity_digest="identity:api-ingest-pages",
            parameter_digest="params:api-ingest-pages",
        )
        authorization = executor.authorize_bound(
            _binding(actuator, operation_name=INGEST_OPERATION), preview, _principal()
        )
        started = executor.begin_bound(
            _binding(actuator, operation_name=INGEST_OPERATION), preview, authorization, object()
        )
        assert started.operation_id is not None
        session_ids = [f"chatgpt:{index:05d}" for index in range(4)]
        audit.append_ingest_session_id_page(started.operation_id, 0, tuple(session_ids))
        return started.operation_id, session_ids

    operation_id, session_ids = await run_archive_fixture_write(tmp_path, prepare)
    digest = ingest_session_ids_digest(session_ids)
    summary = {
        "enumeration_complete": True,
        "parse_projection_known": True,
        "processed_session_ids": [],
        "processed_session_id_pages_ref": operation_id,
        "processed_session_id_page_count": 1,
        "processed_session_ids_digest": digest,
        "processed_message_count": 4,
        "changed_session_count": 4,
        "changed_message_count": 4,
        "confirmed_raw_count": 1,
        "unresolved_raw_count": 0,
    }
    envelope: dict[str, object] = {"outcome": "completed", "result": {"historical_receipt": {"summary": summary}}}

    class FakeClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def operation_to_completion(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return envelope

    monkeypatch.setattr("polylogue.daemon_client.DaemonClient", FakeClient)
    source = tmp_path / "fixture.json"
    source.write_text("{}", encoding="utf-8")
    archive = Polylogue(archive_root=tmp_path)
    try:
        result = await archive.parse_file(source, source_name="chatgpt")
        assert result.processed_ids == set(session_ids)
        assert result.counts["sessions"] == 4
        assert result.changed_counts["sessions"] == 4

        summary["processed_session_id_page_count"] = 2
        with pytest.raises(ValueError, match="page count differs"):
            await archive.parse_file(source, source_name="chatgpt")
        summary["processed_session_id_page_count"] = 1
        summary["processed_session_ids_digest"] = "a" * 64
        with pytest.raises(ValueError, match="differ from terminal receipt"):
            await archive.parse_file(source, source_name="chatgpt")
    finally:
        await archive.close()


def test_ingest_insight_pages_replay_and_reject_missing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.operations import machine_receipts

    audit = _audit(tmp_path)
    actuator = _IngestPageActuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "replayed-insight-page-token")
    binding = _binding(actuator, operation_name=INGEST_OPERATION)
    preview = executor.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:replayed-insight-page",
        archive_identity_digest="identity:replayed-insight-page",
        parameter_digest="params:replayed-insight-page",
    )
    authorization = executor.authorize_bound(binding, preview, _principal())
    started = executor.begin_bound(binding, preview, authorization, object())
    assert started.operation_id is not None
    pages = [
        IngestInsightPageHistoricalReceipt(
            ordinal=ordinal,
            targets=[
                InsightTargetHistoricalReceipt(
                    target_ref=f"session:fixture-{ordinal}",
                    disposition="published",
                    certified_counts=InsightCertifiedCountsHistorical(profiles=1),
                    publication_known_committed=True,
                )
            ],
        )
        for ordinal in range(2)
    ]
    audit.append_ingest_insight_page(started.operation_id, pages[0])
    original_phase = AuditContinuityCoordinator._phase

    def interrupt_page(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "append_ingest_insight_page" and phase == "after_source_prepare":
            raise RuntimeError("crash after insight page prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_page)
    with pytest.raises(RuntimeError, match="crash after insight page prepare"):
        audit.append_ingest_insight_page(started.operation_id, pages[1])
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)
    restarted = AuditRepository.for_archive_root(tmp_path)
    restarted.reconcile_continuity()
    monkeypatch.setattr(machine_receipts, "MAX_MACHINE_RECEIPT_PAGES", 1)
    receipt = IngestHistoricalReceiptV2(
        source_generation_id="generation:fixture",
        final_sequence=1,
        input_count=1,
        input_pages_ref="operation:fixture",
        input_page_count=1,
        input_pages_digest=ingest_input_pages_digest(
            [
                IngestInputPageHistoricalReceipt.from_items(
                    0,
                    [
                        IngestInputHistoricalReceipt(
                            source_item_id="source-item:fixture",
                            logical_coordinate="fixture.json",
                            denominator=1,
                            raw_ids=["raw:fixture"],
                        )
                    ],
                )
            ]
        ),
        insight_pages_ref=started.operation_id,
        insight_page_count=2,
        insight_pages_digest=ingest_insight_pages_digest(pages),
        summary=IngestTerminalSummaryHistorical(
            enumeration_complete=True,
            source_complete=True,
            confirmed_raw_count=1,
            unresolved_raw_count=0,
            profile_targets_observed=2,
        ),
    )
    assert restarted.resolve_ingest_insight_pages(receipt) == pages
    assert (
        restarted.read_ingest_insight_pages(
            started.operation_id,
            page_count=2,
            target_count=2,
            digest=ingest_insight_pages_digest(pages),
        )
        == pages
    )
    restarted.append_ingest_insight_page(started.operation_id, pages[1])
    with pytest.raises(ValueError, match="page count differs"):
        restarted.read_ingest_insight_pages(
            started.operation_id,
            page_count=3,
            target_count=2,
            digest=ingest_insight_pages_digest(pages),
        )
    with pytest.raises(ValueError, match="differ from terminal receipt"):
        restarted.read_ingest_insight_pages(
            started.operation_id,
            page_count=2,
            target_count=2,
            digest="a" * 64,
        )


def test_ingest_input_raw_page_replays_and_verifies_unresolved_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(tmp_path)
    actuator = _IngestPageActuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "replayed-input-raw-page-token")
    binding = _binding(actuator, operation_name=INGEST_OPERATION)
    preview = executor.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:replayed-input-raw-page",
        archive_identity_digest="identity:replayed-input-raw-page",
        parameter_digest="params:replayed-input-raw-page",
    )
    authorization = executor.authorize_bound(binding, preview, _principal())
    started = executor.begin_bound(binding, preview, authorization, object())
    assert started.operation_id is not None
    page = IngestInputRawPageHistoricalReceipt(
        source_item_id="source-item:zip",
        ordinal=0,
        raws=[IngestInputRawMemberHistorical(raw_id=f"raw:{index}", unresolved=index == 3) for index in range(4)],
    )
    original_phase = AuditContinuityCoordinator._phase

    def interrupt_page(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "append_ingest_input_raw_page" and phase == "after_source_prepare":
            raise RuntimeError("crash after input raw page prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_page)
    with pytest.raises(RuntimeError, match="crash after input raw page prepare"):
        audit.append_ingest_input_raw_page(started.operation_id, page)
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)
    restarted = AuditRepository.for_archive_root(tmp_path)
    restarted.reconcile_continuity()
    assert restarted.read_ingest_input_raw_pages(
        started.operation_id,
        source_item_id="source-item:zip",
        page_count=1,
        raw_count=4,
        unresolved_count=1,
        digest=ingest_input_raw_pages_digest([page]),
    ) == [page]
    restarted.append_ingest_input_raw_page(started.operation_id, page)
    with pytest.raises(ValueError, match="page count differs"):
        restarted.read_ingest_input_raw_pages(
            started.operation_id,
            source_item_id="source-item:zip",
            page_count=2,
            raw_count=4,
            unresolved_count=1,
            digest=ingest_input_raw_pages_digest([page]),
        )
    with pytest.raises(ValueError, match="differ from terminal receipt"):
        restarted.read_ingest_input_raw_pages(
            started.operation_id,
            source_item_id="source-item:zip",
            page_count=1,
            raw_count=4,
            unresolved_count=0,
            digest=ingest_input_raw_pages_digest([page]),
        )


def test_ingest_v2_input_pages_are_ordered_and_audit_owned(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    actuator = _IngestPageActuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "ingest-v2-page-token")
    binding = _binding(actuator, operation_name=INGEST_OPERATION)
    preview = executor.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:v2-page",
        archive_identity_digest="identity:v2-page",
        parameter_digest="params:v2-page",
    )
    authorization = executor.authorize_bound(binding, preview, _principal())
    started = executor.begin_bound(binding, preview, authorization, object())
    assert started.operation_id is not None
    pages = [
        IngestInputPageHistoricalReceipt.from_items(
            ordinal,
            [
                IngestInputHistoricalReceipt(
                    source_item_id=f"item:{number}",
                    logical_coordinate=f"input:{number:04d}",
                    denominator=1,
                    raw_ids=[f"raw:{number}"],
                )
                for number in range(ordinal * 256, min((ordinal + 1) * 256, 257))
            ],
        )
        for ordinal in range(2)
    ]
    audit.append_ingest_input_page(started.operation_id, pages[0])
    audit.append_ingest_input_page(started.operation_id, pages[0])
    with pytest.raises(ValueError, match="conflicts"):
        audit.append_ingest_input_page(
            started.operation_id,
            IngestInputPageHistoricalReceipt.from_items(
                0,
                [
                    IngestInputHistoricalReceipt(
                        source_item_id="other", logical_coordinate="input:0000", denominator=0, raw_ids=[]
                    )
                ],
            ),
        )
    with pytest.raises(ValueError, match="contiguous"):
        audit.append_ingest_input_page(started.operation_id, pages[1].model_copy(update={"ordinal": 3}))
    audit.append_ingest_input_page(started.operation_id, pages[1])
    root = IngestHistoricalReceiptV2(
        source_generation_id="generation:v2",
        final_sequence=1,
        input_count=257,
        input_pages_ref=started.operation_id,
        input_page_count=2,
        input_pages_digest=ingest_input_pages_digest(pages),
        summary=IngestTerminalSummaryHistorical(
            enumeration_complete=True,
            source_complete=True,
            confirmed_raw_count=257,
            unresolved_raw_count=0,
            profile_targets_observed=0,
        ),
    )
    assert audit.read_ingest_input_pages(root) == pages
    executor.finalize_bound(
        started,
        receipt=MutationReceipt(
            operation=started.plan.operation,
            plan_hash=started.plan.plan_hash,
            status="applied",
            target_refs=started.plan.target_refs,
            affected_count=1,
            detail=None,
            receipt_ref=None,
            applied_at=started.plan.prepared_at,
            historical_receipt=root,
        ),
    )
    (tmp_path / "source.db").rename(tmp_path / "source.unavailable")
    (tmp_path / "index.db").unlink(missing_ok=True)
    assert audit.historical_machine_receipt(started.operation_id) == root
    assert list(audit.iter_ingest_input_pages(root)) == pages
    with pytest.raises(ValueError, match="missing"):
        audit.read_ingest_input_pages(root.model_copy(update={"input_page_count": 3}))
    with pytest.raises(ValueError, match="missing"):
        audit.read_ingest_input_pages(root.model_copy(update={"input_pages_ref": "operation:foreign"}))
    with pytest.raises(ValueError, match="terminal root"):
        audit.read_ingest_input_pages(root.model_copy(update={"input_pages_digest": "0" * 64}))


def test_pending_command_of_an_undeclared_kind_is_a_typed_refusal(tmp_path: Path) -> None:
    """A prepared command this runtime does not declare refuses reconciliation by name.

    Only another runtime can have prepared it, so replay would guess its effect
    and clearing it would drop an effect that may have committed. Anti-vacuity:
    raise a bare ``RuntimeError`` from ``_replay_domain_mutation`` again and the
    ``pytest.raises`` below fails; clear the pending entry instead and the
    final assertion fails.
    """
    audit = _audit(tmp_path)
    coordinator = AuditContinuityCoordinator(tmp_path)
    coordinator._prepare(AuditMutation("retired_kind", "mutation:retired", 0, {}))

    with pytest.raises(AuditContinuityUnknownMutationError) as refusal:
        audit.reconcile_continuity()

    assert refusal.value.kind == "retired_kind"
    assert coordinator._pending() is not None


def test_ingest_refusal_pages_resolve_every_named_refusal(tmp_path: Path) -> None:
    """More refusals than one page are all retained and resolved, none dropped.

    Anti-vacuity: truncate the refusal enumeration (the removed 256 cap) and
    ``resolve_ingest_refusals`` returns fewer refusals than the receipt counts,
    which it refuses as differing from the terminal receipt.
    """
    from polylogue.operations.machine_receipts import (
        MAX_PAGE_ITEMS,
        IngestRefusalPageHistoricalReceipt,
        IngestRefusedMembershipHistorical,
        ingest_refusal_pages_digest,
    )

    audit = _audit(tmp_path)
    actuator = _IngestPageActuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "refusal-page-token")
    binding = _binding(actuator, operation_name=INGEST_OPERATION)
    preview = executor.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:refusal-page",
        archive_identity_digest="identity:refusal-page",
        parameter_digest="params:refusal-page",
    )
    authorization = executor.authorize_bound(binding, preview, _principal())
    started = executor.begin_bound(binding, preview, authorization, object())
    operation_id = started.operation_id
    assert operation_id is not None
    refusals = [
        IngestRefusedMembershipHistorical(
            logical_source_key=f"codex-session:{index:05d}", raw_id=f"raw:{index:05d}", reason="did not parse"
        )
        for index in range(MAX_PAGE_ITEMS + 1)
    ]
    pages = [
        IngestRefusalPageHistoricalReceipt(ordinal=0, refusals=refusals[:MAX_PAGE_ITEMS]),
        IngestRefusalPageHistoricalReceipt(ordinal=1, refusals=refusals[MAX_PAGE_ITEMS:]),
    ]
    for page in pages:
        audit.append_ingest_refusal_page(operation_id, page)
    audit.append_ingest_refusal_page(operation_id, pages[1])

    def receipt(digest: str) -> IngestHistoricalReceiptV2:
        return IngestHistoricalReceiptV2(
            source_generation_id="generation:fixture",
            final_sequence=1,
            input_count=1,
            input_pages_ref=operation_id,
            input_page_count=1,
            input_pages_digest="0" * 64,
            summary=IngestTerminalSummaryHistorical(
                enumeration_complete=True,
                source_complete=False,
                confirmed_raw_count=0,
                unresolved_raw_count=0,
                profile_targets_observed=0,
                refused_membership_count=len(refusals),
                refused_membership_pages_ref=operation_id,
                refused_membership_page_count=len(pages),
                refused_memberships_digest=digest,
            ),
        )

    assert audit.resolve_ingest_refusals(receipt(ingest_refusal_pages_digest(pages))) == refusals
    with pytest.raises(ValueError, match="differ from terminal receipt"):
        audit.resolve_ingest_refusals(receipt("a" * 64))
    with pytest.raises(ValueError, match="conflicts with durable page"):
        audit.append_ingest_refusal_page(
            operation_id, IngestRefusalPageHistoricalReceipt(ordinal=1, refusals=refusals[:1])
        )


def test_recovery_needing_an_unservable_tier_is_deferred_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A schema refusal while recovering leaves the run for the next attempt.

    Anti-vacuity: let ``SchemaRefusalError`` fall into the generic handler in
    ``resolve_interrupted_operation`` and the run is terminalized as
    ``recovery_replay_failed`` although nothing was tried.
    """
    from polylogue.core.errors import SchemaSkewError

    actuator = _Actuator()

    def skewed(_handles: ReplayHandles, _plan: MutationPlan) -> RecoveryResolution:
        raise SchemaSkewError("index", "identity:new", "identity:old")

    monkeypatch.setattr(actuator, "recover", skewed)
    _register_fixture(monkeypatch, actuator)
    _audit_repo, operation_id = _dead_nonterminal_operation(tmp_path, actuator)

    recover_on_admitted_owner(tmp_path)

    status, _reason, _targets = _run_state(tmp_path, operation_id)
    assert status == "interrupted"


def test_a_request_overlapping_deferred_recovery_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """New work on targets whose interrupted run could not be resolved waits for it.

    Anti-vacuity: ignore the deferred ids returned to
    ``_resolve_overlapping_operations`` and the new request applies before the
    older interrupted one.
    """
    from polylogue.core.errors import SchemaSkewError

    actuator = _Actuator()

    def skewed(_handles: ReplayHandles, _plan: MutationPlan) -> RecoveryResolution:
        raise SchemaSkewError("index", "identity:new", "identity:old")

    monkeypatch.setattr(actuator, "recover", skewed)
    _register_fixture(monkeypatch, actuator)
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)

    with pytest.raises(RecoveryBlockedError, match="cannot resolve yet"):
        _retry(audit, actuator, "after-deferral-token")

    assert actuator.calls == 0
    # Left for a later attempt: not terminalized, still overlapping its targets.
    assert _run_state(tmp_path, operation_id)[0] in {"running", "interrupted"}


def test_paged_machine_batch_stages_then_completes_and_survives_recovery(tmp_path: Path) -> None:
    """A batch accepted in pages is one durable request: running while staged,
    complete at its final page, and its pages cannot be skipped or repeated.

    Anti-vacuity: insert a new machine request per page, or skip the staged
    kind, and the second page raises ``MachineRequestRecoveredError`` or the
    staged request reads as completed.
    """
    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit)
    previews = tuple(
        executor.prepare_bound(
            _binding(actuator),
            object(),
            _principal(),
            archive_instance_id="archive:fixture",
            archive_identity_digest="identity:fixture",
            parameter_digest=f"params:{i}",
        )
        for i in range(3)
    )
    authorizations = tuple(executor.authorize_bound(_binding(actuator), preview, _principal()) for preview in previews)
    refs = tuple(str(auth.authorization_id) for auth in authorizations)
    binding = MachineRequestBinding("identity:fixture", "request:paged", "actor:test", "d" * 64, "mutation.fixture")

    with audit.bind_machine_request(binding, transition="accept_execution_batch", page=(0, False)):
        audit.accept_execution_batch(refs[:2], _principal())
    staged = audit.machine_request(binding)
    assert staged is not None and staged["artifact_kind"] == "execution-batch-pages"
    assert machine_request_state(audit, staged)["outcome"] == "running"
    with audit.bind_machine_request(binding, transition="accept_execution_batch", page=(1, True)):
        with pytest.raises(MachineRequestRecoveredError):
            audit.accept_execution_batch(refs[2:], _principal())

    recovered = AuditRepository.for_archive_root(tmp_path)
    recovered.reconcile_continuity()
    with recovered.bind_machine_request(binding, transition="accept_execution_batch", page=(2, True)):
        recovered.accept_execution_batch(refs[2:], _principal())
    complete = recovered.machine_request(binding)
    assert complete is not None and complete["artifact_kind"] == "execution-batch"
    assert complete["part_count"] == 3
    assert [part["ordinal"] for part in recovered.machine_parts(binding)] == [0, 1, 2]
    assert [part["authorization_ref"] for part in recovered.machine_parts(binding)] == list(refs)
    with recovered.bind_machine_request(binding, transition="accept_execution_batch", page=(3, True)):
        with pytest.raises(MachineRequestRecoveredError):
            recovered.accept_execution_batch(refs[:1], _principal())


def test_startup_fences_a_half_accepted_paged_batch(tmp_path: Path) -> None:
    """Anti-vacuity: leave a staged request unfenced at startup and it reads as
    running forever, so a follower never reaches a terminal outcome."""
    from polylogue.operations.daemon_protocol import AcceptedOperationReference

    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit)
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:fixture",
        archive_identity_digest="identity:fixture",
        parameter_digest="params:fence",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    binding = MachineRequestBinding("identity:fixture", "request:fence", "actor:test", "e" * 64, "mutation.fixture")
    with audit.bind_machine_request(binding, transition="accept_execution_batch", page=(0, False)):
        audit.accept_execution_batch((str(authorization.authorization_id),), _principal())

    assert audit.fence_staged_machine_pages() == 1
    record = audit.machine_request(binding)
    assert record is not None
    assert machine_request_state(audit, record)["outcome"] == "interrupted"
    assert audit.fence_staged_machine_pages() == 0
    reference = AcceptedOperationReference.from_record({**record, "part_count": 41})
    assert reference.part_count == 41


def test_a_failing_later_page_terminalizes_the_staged_request(tmp_path: Path) -> None:
    """Anti-vacuity: leave a staged request unstopped when a later page raises
    and it reads as running until the next restart."""
    from polylogue.operations.daemon_mutations import _fenced_on_failure

    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit)
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:fixture",
        archive_identity_digest="identity:fixture",
        parameter_digest="params:failing",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    binding = MachineRequestBinding("identity:fixture", "request:failing", "actor:test", "f" * 64, "mutation.fixture")
    with pytest.raises(RuntimeError, match="second page"), _fenced_on_failure(audit, binding):
        with audit.bind_machine_request(binding, transition="accept_execution_batch", page=(0, False)):
            audit.accept_execution_batch((str(authorization.authorization_id),), _principal())
        raise RuntimeError("second page failed")
    record = audit.machine_request(binding)
    assert record is not None and record["stop_reason"] == "refused"
    assert machine_request_state(audit, record)["outcome"] == "interrupted"


def test_cancelling_a_staged_authorization_batch_between_pages_revokes_it(tmp_path: Path) -> None:
    """Anti-vacuity: stop checking cancellation between pages and the second
    page is accepted; leave staged authorizations active on a fence and the
    first page's authorization stays usable."""
    from types import SimpleNamespace

    from polylogue.archive.query.execution_control import QueryCancelledError
    from polylogue.operations.audit import MACHINE_PAGE_PARTS
    from polylogue.operations.daemon_mutations import _fenced_on_failure, _page_bounds

    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit)
    previews = [
        executor.prepare_bound(
            _binding(actuator),
            object(),
            _principal(),
            archive_instance_id="archive:fixture",
            archive_identity_digest="identity:fixture",
            parameter_digest=f"params:paged-{index}",
        )
        for index in range(2)
    ]
    binding = MachineRequestBinding("identity:fixture", "request:paged", "actor:test", "c" * 64, "mutation.fixture")
    context = cast(Any, SimpleNamespace(runtime=SimpleNamespace(stop_reason=lambda _request: "cancelled")))
    accepted_pages = 0
    with pytest.raises(QueryCancelledError, match="cancelled"), _fenced_on_failure(audit, binding):
        for offset, _end, final in _page_bounds(
            2 * MACHINE_PAGE_PARTS, 0, request=cast(Any, None), context=context, audit=audit, binding=binding
        ):
            preview = previews[accepted_pages]
            # Issued unpersisted, as the daemon handler issues them; the batch publishes it.
            authorization = OperationExecutor().authorize_bound(_binding(actuator), preview, _principal())
            with audit.bind_machine_request(binding, transition="issue_authorization_batch", page=(offset, final)):
                audit.issue_authorization_batch((preview,), _principal(), (authorization,))
            accepted_pages += 1
    assert accepted_pages == 1
    record = audit.machine_request(binding)
    assert record is not None and record["stop_reason"] == "cancelled"
    assert machine_request_state(audit, record)["outcome"] == "cancelled"
    refs = [str(part["artifact_ref"]) for part in audit.machine_parts(binding)]
    with audit._connection() as conn:
        states = {
            str(row[0])
            for row in conn.execute(
                f"SELECT state FROM operation_authorizations WHERE authorization_id IN ({','.join('?' * len(refs))})",
                refs,
            )
        }
    assert refs and states == {"revoked"}


class _AuditClock:
    """``time`` for the audit module, with a settable wall clock."""

    def __init__(self, ms: int) -> None:
        self.ms = ms

    def time(self) -> float:
        return self.ms / 1000

    def __getattr__(self, name: str) -> object:
        import time as real_time

        return getattr(real_time, name)


@pytest.mark.parametrize("idle_ms", [10_000, 61_000])
def test_a_progressing_paged_handshake_keeps_its_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, idle_ms: int
) -> None:
    """Authority expires while a delete handshake sits idle, not while it pages.

    The previews expire 60 s after the preview request is accepted, and its
    second page lands 50 s later. Anti-vacuity: judge authorization by wall
    time and a handshake that moves on 10 s after its last preview page is
    refused as expired; judge it by the preview's acceptance regardless of
    idle time and one left idle 61 s is still authorized.
    """
    from polylogue.operations import audit as audit_module
    from polylogue.operations.audit import MachineRequestBinding as Binding

    audit = _audit(tmp_path)
    clock = _AuditClock(1_000_000_000)
    monkeypatch.setattr(audit_module, "time", clock)
    actuator = _Actuator()
    start = clock.ms
    preparer = OperationExecutor(now_ms=lambda: clock.ms)
    plans = [
        preparer.prepare_bound(
            _binding(actuator),
            object(),
            _principal(),
            archive_instance_id="archive:fixture",
            archive_identity_digest="identity:fixture",
            parameter_digest=f"params:handshake-{index}",
            expires_at_ms=start + 60_000,
        ).plan
        for index in range(2)
    ]
    previewing = Binding("identity:fixture", "request:preview", "actor:test", "a" * 64, "mutation.fixture")
    for offset, plan in enumerate(plans):
        with audit.bind_machine_request(
            previewing, transition="create_preview_batch", page=(offset, offset == len(plans) - 1)
        ):
            audit.create_preview_batch((plan,), _principal())
        clock.ms += 50_000
    refs = [str(part["artifact_ref"]) for part in audit.machine_parts(previewing)]
    previews = tuple(audit.preview_for_principal(ref, _principal()) for ref in refs)

    clock.ms = start + 50_000 + idle_ms
    authorizing = Binding("identity:fixture", "request:authorize", "actor:test", "b" * 64, "mutation.fixture")
    as_of_ms = audit.handshake_as_of_ms(authorizing, preview_refs=tuple(refs))
    if idle_ms > audit_module.HANDSHAKE_IDLE_MS:
        assert as_of_ms == clock.ms
        with pytest.raises(TokenExpiredError):
            OperationExecutor(now_ms=lambda: as_of_ms).authorize_bound(_binding(actuator), previews[0], _principal())
        return
    assert as_of_ms == start
    authorizations = tuple(
        OperationExecutor(now_ms=lambda: as_of_ms).authorize_bound(_binding(actuator), preview, _principal())
        for preview in previews
    )
    with audit.bind_machine_request(authorizing, transition="issue_authorization_batch", page=(0, True)):
        audit.issue_authorization_batch(previews, _principal(), authorizations)
    authorization_refs = tuple(str(part["artifact_ref"]) for part in audit.machine_parts(authorizing))

    clock.ms += 30_000
    executing = Binding("identity:fixture", "request:execute", "actor:test", "c" * 64, "mutation.fixture")
    with audit.bind_machine_request(executing, transition="accept_execution_batch", page=(0, True)):
        audit.accept_execution_batch(authorization_refs, _principal())
    record = audit.machine_request(executing)
    assert record is not None and record["artifact_kind"] == "execution-batch"


def test_recovery_discovery_does_not_bind_the_pre_begin_lease(tmp_path: Path) -> None:
    from polylogue.core.write_lease import current_write_lease
    from tests.infra.archive_templates import run_archive_fixture_prepare

    bootstrap_archive_root(tmp_path)

    def discover() -> None:
        audit = AuditRepository.for_archive_root(tmp_path)
        executor = OperationExecutor(audit=audit, archive_root=tmp_path)
        assert current_write_lease() is None
        executor._resolve_dead_operations(resolver_actor_ref=_principal().actor_ref, prepared_excision_only=True)
        assert current_write_lease() is None

    asyncio.run(run_archive_fixture_prepare(discover))


def test_recovery_discovery_borrows_the_actual_coordinated_audit_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap_archive_root(tmp_path)
    observed: list[str] = []

    def execute() -> None:
        audit = AuditRepository.for_archive_root(tmp_path)
        executor = OperationExecutor(audit=audit, archive_root=tmp_path)
        original = audit._bind_machine_result

        def bind_result(conn: sqlite3.Connection, mutation: AuditMutation, result: object) -> None:
            assert audit._coordinated_connection is conn
            executor._resolve_dead_operations(resolver_actor_ref=_principal().actor_ref, prepared_excision_only=True)
            with audit.recovery_discovery_read(), audit._connection() as discovered:
                assert discovered is conn
            observed.append(mutation.kind)
            original(conn, mutation, result)

        monkeypatch.setattr(audit, "_bind_machine_result", bind_result)
        from polylogue.storage.archive_identity import ArchiveIdentity

        instance = audit.ensure_archive_authority(now_ms=1_700_000_000_000)
        identity = ArchiveIdentity.resolve(tmp_path)
        actuator = _Actuator()
        preview = executor.prepare_bound(
            _binding(actuator),
            object(),
            _principal(),
            archive_instance_id=instance,
            archive_identity_digest=identity.authority_identity_digest,
            parameter_digest="params:fixture",
        )
        authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
        receipt = executor.execute_bound(_binding(actuator), preview, authorization, object())
        assert receipt.status == "applied"
        assert actuator.calls == 1

    asyncio.run(run_archive_fixture_write(tmp_path, execute))
    assert "consume_authorization_and_start" in observed


@pytest.mark.parametrize("cancel_waiter", [False, True])
def test_recovery_discovery_settles_original_reader_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_waiter: bool
) -> None:
    from polylogue.core.compute import BoundedComputeAdapter, DaemonOperationCancelled
    from polylogue.core.write_lease import current_write_lease
    from polylogue.storage.sqlite.audit_continuity import AuditContinuityPendingError

    bootstrap_archive_root(tmp_path)
    audit = AuditRepository.for_archive_root(tmp_path)
    original_lock = audit._continuity._execution_lock
    contended = threading.Event()
    rendezvous = threading.Event()

    class ObservedLock:
        def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
            acquired = original_lock.acquire(blocking, timeout)
            if not acquired and blocking:
                contended.set()
                rendezvous.set()
            return acquired

        def release(self) -> None:
            original_lock.release()

    monkeypatch.setattr(audit._continuity, "_execution_lock", ObservedLock())
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=1)

    def discover() -> tuple[int, bool]:
        assert current_write_lease() is None
        with pytest.raises(AuditContinuityPendingError):
            with audit.settled_machine_read():
                raise AssertionError("completion read waited behind a held continuity reader")
        with audit.recovery_discovery_read():
            rows = audit.orphaned_operations()
        return len(rows), current_write_lease() is None

    submitted = None
    try:
        with audit.settled_machine_read():
            submitted = adapter.submit(discover)
            submitted.future.add_done_callback(lambda _future: rendezvous.set())
            rendezvous.wait()
            if submitted.future.done():
                submitted.future.result()
            assert contended.is_set()
            assert not submitted.future.done()
            if cancel_waiter:
                submitted.cancellation.cancel()
                with pytest.raises(DaemonOperationCancelled):
                    submitted.future.result()
                assert submitted.future.done()
            else:
                assert current_write_lease() is None
        if not cancel_waiter:
            assert submitted.future.result() == (0, True)
        assert original_lock.acquire(blocking=False)
        original_lock.release()
    finally:
        if submitted is not None and not submitted.future.done():
            submitted.cancellation.cancel()
        adapter.shutdown(wait=True)
    assert not any(thread.is_alive() for thread in adapter.executor._threads)


def test_recovery_discovery_preserves_original_pending_head_refusal(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.audit_continuity import AuditContinuityPendingError
    from tests.infra.archive_templates import run_archive_fixture_prepare
    from tests.infra.audit_completion import make_source_completion_control

    _root, audit, payload, install = make_source_completion_control(tmp_path)
    install(payload)

    def discover() -> None:
        with pytest.raises(AuditContinuityPendingError):
            with audit.recovery_discovery_read():
                raise AssertionError("an unpromoted Source command was admitted")

    asyncio.run(run_archive_fixture_prepare(discover))

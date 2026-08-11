from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from pydantic import BaseModel

from polylogue.operations.audit import AuditRepository
from polylogue.operations.bindings import OperationBinding
from polylogue.operations.mutation_transaction import (
    AuditFinalizationError,
    CapabilityDeniedError,
    ConfirmationStrength,
    DestructiveClass,
    MutationPlan,
    MutationPrincipal,
    MutationReceipt,
    OperationExecutor,
    PlanStaleError,
    TargetAuthorityPolicy,
    TargetDurability,
    build_plan,
)
from polylogue.operations.specs import OperationKind, OperationSpec
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator, AuditMutation


@dataclass
class _Actuator:
    operation: str = "mutate-fixture"
    operation_version: int = 1
    changed: bool = False
    calls: int = 0
    crash: bool = False
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, _args: object) -> MutationPlan:
        target = "session:changed" if self.changed else "session:fixture"
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(target,),
            affected_tiers=("user",),
            reversible=True,
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
            affected_count=1,
            detail=None,
            receipt_ref=None,
            applied_at="now",
        )


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


def _binding(
    actuator: _Actuator, *, target_durability: TargetDurability = "derived"
) -> OperationBinding[object, object]:
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
                destructive_class="reversible",
                required_confirmation="role_only",
                allowed_durabilities=(target_durability,),
                allowed_recovery=("none",),
            ),
        ),
        affected_tiers=("user",),
    )
    return OperationBinding(spec, actuator)


def _principal() -> MutationPrincipal:
    return MutationPrincipal("actor:test", frozenset({"archive.fixture.write"}), "internal", "system")


def _audit(tmp_path: Path) -> AuditRepository:
    initialize_active_archive_root(tmp_path)
    return AuditRepository.for_archive_root(tmp_path)


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
    initialize_active_archive_root(tmp_path)
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
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)

    AuditRepository.for_archive_root(tmp_path).reconcile_continuity()
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT status FROM operation_runs").fetchone() == ("completed",)
    with sqlite3.connect(tmp_path / "source.db") as source:
        command = source.execute("SELECT pending_payload_json FROM audit_continuity_control").fetchone()[0]
    assert command is None


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

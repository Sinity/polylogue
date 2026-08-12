from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from pydantic import BaseModel

from polylogue.operations.audit import (
    AuditRepository,
    _attempt_owner_is_live,
    _attempt_owner_liveness,
    _current_process_attempt_owner,
    token_sha256,
)
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
    target_refs: tuple[str, ...] = ("session:fixture",)
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, _args: object) -> MutationPlan:
        targets = ("session:changed",) if self.changed else self.target_refs
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=targets,
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
            affected_count=len(plan.target_refs),
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
    digest_as_bearer = replace(authorization, token=f"sha256:{token_sha256('raw-secret-token')}")
    with pytest.raises(ValueError, match="does not match preview"):
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


def test_production_factory_does_not_abandon_a_live_same_process_attempt(tmp_path: Path) -> None:
    """A second composition-root call recognizes the first executor's owner."""
    initialize_active_archive_root(tmp_path)
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
    initialize_active_archive_root(tmp_path)
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


@pytest.mark.parametrize("owner_id", [None, "external:unverifiable"])
def test_recovery_preserves_attempts_with_unproven_owners(tmp_path: Path, owner_id: str | None) -> None:
    """Legacy or externally-owned attempts stay running until their owner is proven dead."""

    initialize_active_archive_root(tmp_path)
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


def test_reconciliation_resolves_the_full_unknown_atomic_batch(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    actuator = _Actuator(target_refs=("session:first", "session:second"))
    executor = OperationExecutor(audit=audit, token_factory=lambda: "reconcile-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:reconcile",
        archive_identity_digest="identity:reconcile",
        parameter_digest="params:reconcile",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    operation_id = audit.consume_authorization_and_start(preview, authorization)
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        conn.execute(
            "UPDATE operation_attempts SET worker_id = 'pid:999999999:0' WHERE operation_id = ?", (operation_id,)
        )
        conn.commit()
    audit.recover_abandoned_attempts()

    audit.reconcile_attempt(operation_id, outcome="applied", domain_receipt_ref="receipt:reconciled")

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT status, affected_count, unknown_count FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("completed", 2, 0)
        assert conn.execute(
            "SELECT state, domain_receipt_ref FROM operation_targets WHERE operation_id = ? ORDER BY ordinal",
            (operation_id,),
        ).fetchall() == [("applied", "receipt:reconciled"), ("applied", "receipt:reconciled")]


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

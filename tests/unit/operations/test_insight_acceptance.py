"""Exact-page authority tests for daemon-owned insight maintenance."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from polylogue.operations.audit import AuditRepository, MachineRequestBinding, MachineRequestRecoveredError
from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.insight_acceptance import (
    AcceptedInsightPart,
    AcceptedInsightTarget,
    InsightAcceptance,
    InsightScopeKind,
    build_insight_page_context,
    insight_manifest_digest,
)
from polylogue.operations.mutation_actuators import InsightsRebuildActuator, InsightsRebuildArgs
from polylogue.operations.mutation_transaction import (
    AuthorizationMismatchError,
    MutationPlan,
    MutationPrincipal,
    MutationTarget,
    MutationTransactionError,
    OperationExecutor,
    build_typed_plan,
)
from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator, AuditMutation
from tests.infra.archive_templates import bootstrap_archive_root


def _principal() -> MutationPrincipal:
    return MutationPrincipal(
        "actor:daemon",
        frozenset({"archive.rebuild_insights"}),
        "api",
        "daemon",
    )


def _target(session_id: str) -> MutationTarget:
    ref = f"session:{session_id}"
    return MutationTarget(
        kind="session",
        ref=ref,
        policy_key="insights-rebuild",
        identity_digest=hashlib.sha256(ref.encode()).hexdigest(),
        effect_identity=f"mutate-rebuild-insights:{ref}",
        durability="derived",
        recovery="rebuild",
    )


def _page(
    *,
    scope_kind: InsightScopeKind = "full",
    ordinal: int,
    count: int,
    digest: str,
    previous_preview_ref: str | None,
    targets: tuple[AcceptedInsightTarget, ...],
) -> MutationPlan:
    return build_typed_plan(
        operation="mutate-rebuild-insights",
        operation_version=1,
        archive_instance_id="archive:fixture",
        archive_identity_digest="a" * 64,
        targets=tuple(_target(item.target_ref.removeprefix("session:")) for item in targets),
        affected_tiers=("index",),
        parameter_digest=f"fixture:{ordinal}",
        required_capabilities=("archive.rebuild_insights",),
        destructive_class="maintenance",
        required_confirmation="role_only",
        prepared_at_ms=1,
        expires_at_ms=9_999_999_999_999,
        context=build_insight_page_context(
            scope_kind=scope_kind,
            index_generation="index-generation:/archive/fixture/index.db",
            recipe_version="2",
            ordinal=ordinal,
            page_count=count,
            manifest_digest=digest,
            previous_preview_ref=previous_preview_ref,
            targets=targets,
        ),
    )


def test_seal_preserves_over_ten_thousand_exact_targets_without_late_rediscovery(tmp_path: Path) -> None:
    """Replacing page refs with a live full scan would add the late target and make this red."""

    bootstrap_archive_root(tmp_path)
    audit = AuditRepository.for_archive_root(tmp_path)
    principal = _principal()
    page_count = 41
    page_targets = tuple(
        tuple(AcceptedInsightTarget(f"session:fixture-{page * 256 + item}", "required") for item in range(256))
        for page in range(page_count)
    )
    provisional = tuple(
        AcceptedInsightPart(
            preview_ref=f"pending:{page}",
            authorization_ref=f"pending-auth:{page}",
            plan_hash="0" * 64,
            ordinal=page,
            page_count=page_count,
            manifest_digest="0" * 64,
            previous_preview_ref=None if page == 0 else f"pending:{page - 1}",
            scope_kind="full",
            index_generation="index-generation:/archive/fixture/index.db",
            recipe_version="2",
            targets=page_targets[page],
        )
        for page in range(page_count)
    )
    digest = insight_manifest_digest(provisional)
    previews = []
    authorizations = []
    previous: str | None = None
    executor = OperationExecutor(audit=audit)
    binding = runtime_operation_binding(InsightsRebuildActuator())
    machine = MachineRequestBinding(
        "a" * 64,
        "request:more-than-ten-thousand",
        principal.actor_ref,
        "b" * 64,
        "maintenance.insights.rebuild",
    )
    acceptance = InsightAcceptance(audit, machine, principal)
    for ordinal, targets in enumerate(page_targets):
        plan = _page(
            ordinal=ordinal,
            count=page_count,
            digest=digest,
            previous_preview_ref=previous,
            targets=targets,
        )
        preview = acceptance.stage_preview(plan)
        assert acceptance.stage_preview(plan) == preview
        authorization = acceptance.ensure_staged_authorization(executor, binding, preview)
        assert authorization.authorization_id is not None
        previews.append(preview)
        authorizations.append(authorization)
        previous = preview.preview_ref

    accepted = acceptance.seal(
        head_preview_ref=previews[-1].preview_ref,
        page_count=page_count,
        manifest_digest=digest,
        deadline_unix_ms=9_999_999_999_999,
    )
    assert len(accepted) == page_count
    assert sum(len(part.targets) for part in accepted) == 10_496
    assert "session:late-after-seal" not in {target.target_ref for part in accepted for target in part.targets}
    assert [part.authorization_ref for part in accepted] == [str(item.authorization_id) for item in authorizations]
    # A target arriving after acceptance cannot extend the frozen full sweep
    # through the preparation route.  It needs a distinct request and its own
    # immutable manifest rather than an ambient re-scan.
    with pytest.raises(MachineRequestRecoveredError):
        acceptance.stage_preview(
            _page(
                ordinal=page_count,
                count=page_count + 1,
                digest="d" * 64,
                previous_preview_ref=previews[-1].preview_ref,
                targets=(AcceptedInsightTarget("session:late-after-seal", "required"),),
            )
        )
    # Repeating the exact request reloads the sealed parts; it cannot issue a
    # replacement authorization or rediscover the ambient archive population.
    assert (
        InsightAcceptance(audit, machine, principal).seal(
            head_preview_ref=previews[-1].preview_ref,
            page_count=page_count,
            manifest_digest=digest,
            deadline_unix_ms=9_999_999_999_999,
        )
        == accepted
    )


def test_manifest_rejects_a_tampered_predecessor_before_reserving_authority(tmp_path: Path) -> None:
    """Removing predecessor-chain validation would reserve a reordered page set."""

    bootstrap_archive_root(tmp_path)
    audit = AuditRepository.for_archive_root(tmp_path)
    principal = _principal()
    targets = (AcceptedInsightTarget("session:one", "required"),)
    first = AcceptedInsightPart(
        "pending:0",
        "pending-auth:0",
        "0" * 64,
        0,
        2,
        "0" * 64,
        None,
        "explicit",
        "index-generation:/archive/fixture/index.db",
        "2",
        targets,
    )
    second = replace(
        first, preview_ref="pending:1", authorization_ref="pending-auth:1", ordinal=1, previous_preview_ref="wrong"
    )
    digest = insight_manifest_digest((first, second))
    first_plan = _page(ordinal=0, count=2, digest=digest, previous_preview_ref=None, targets=targets)
    executor = OperationExecutor(audit=audit)
    binding = runtime_operation_binding(InsightsRebuildActuator())
    machine = MachineRequestBinding(
        "a" * 64, "request:tampered", principal.actor_ref, "c" * 64, "maintenance.insights.rebuild"
    )
    acceptance = InsightAcceptance(audit, machine, principal)
    first_preview = acceptance.stage_preview(first_plan)
    first_auth = acceptance.ensure_staged_authorization(executor, binding, first_preview)
    assert first_auth.authorization_id is not None
    # Generic executor entry points must not turn a staged bearer token into
    # work before the source-WAL manifest seal reserves its machine ordinal.
    blocked_args = cast(InsightsRebuildArgs, object())
    with pytest.raises(MutationTransactionError, match="sealed accepted machine part"):
        executor.begin_bound(binding, first_preview, first_auth, blocked_args)
    with pytest.raises(MutationTransactionError, match="sealed accepted machine part"):
        executor.execute_bound(binding, first_preview, first_auth, blocked_args)
    with pytest.raises(MutationTransactionError, match="sealed accepted machine part"):
        executor.execute(binding.actuator, first_preview.plan, first_auth, blocked_args)
    second_plan = _page(ordinal=1, count=2, digest=digest, previous_preview_ref="wrong", targets=targets)
    second_preview = acceptance.stage_preview(second_plan)
    second_auth = acceptance.ensure_staged_authorization(executor, binding, second_preview)
    assert second_auth.authorization_id is not None
    with pytest.raises(AuthorizationMismatchError, match="preview|manifest"):
        InsightAcceptance(audit, machine, principal).seal(
            head_preview_ref=second_preview.preview_ref,
            page_count=2,
            manifest_digest=digest,
            deadline_unix_ms=9_999_999_999_999,
        )
    # The two previews are durable staging, but the failed seal has not bound
    # either one-shot authority to executable machine parts.
    staged = audit.machine_parts(machine)
    assert [part["authorization_ref"] for part in staged] == [None, None]
    machine_request = audit.machine_request(machine)
    assert machine_request is not None
    assert machine_request["artifact_kind"] == "insight-preview-pages"


def test_empty_explicit_scope_is_a_sealed_no_effect_part(tmp_path: Path) -> None:
    """Turning explicit [] into a full sweep would make this exact empty part nonempty."""

    bootstrap_archive_root(tmp_path)
    audit = AuditRepository.for_archive_root(tmp_path)
    principal = _principal()
    targets: tuple[AcceptedInsightTarget, ...] = ()
    provisional = AcceptedInsightPart(
        "pending:empty",
        "pending-auth:empty",
        "0" * 64,
        0,
        1,
        "0" * 64,
        None,
        "explicit",
        "index-generation:/archive/fixture/index.db",
        "2",
        targets,
    )
    digest = insight_manifest_digest((provisional,))
    plan = _page(
        scope_kind="explicit",
        ordinal=0,
        count=1,
        digest=digest,
        previous_preview_ref=None,
        targets=targets,
    )
    binding = runtime_operation_binding(InsightsRebuildActuator())
    machine = MachineRequestBinding(
        "a" * 64, "request:empty-explicit", principal.actor_ref, "e" * 64, "maintenance.insights.rebuild"
    )
    acceptance = InsightAcceptance(audit, machine, principal)
    preview = acceptance.stage_preview(plan)
    authorization = acceptance.ensure_staged_authorization(OperationExecutor(audit=audit), binding, preview)
    assert authorization.authorization_id is not None

    accepted = acceptance.seal(
        head_preview_ref=preview.preview_ref,
        page_count=1,
        manifest_digest=digest,
        deadline_unix_ms=9_999_999_999_999,
    )

    assert accepted[0].scope_kind == "explicit"
    assert accepted[0].targets == ()


@pytest.mark.parametrize(
    ("plan_factory", "message"),
    [
        (
            lambda: replace(
                _page(ordinal=0, count=1, digest="f" * 64, previous_preview_ref=None, targets=()),
                operation="mutate-delete-session",
            ),
            "rebuild-insights operation",
        ),
        (
            lambda: replace(
                _page(ordinal=0, count=1, digest="f" * 64, previous_preview_ref=None, targets=()),
                archive_identity_digest="f" * 64,
            ),
            "archive differs",
        ),
    ],
)
def test_stage_refuses_wrong_operation_or_archive_before_wal_prepare(
    tmp_path: Path, plan_factory: Callable[[], object], message: str
) -> None:
    """Dropping operation/archive checks would stage foreign authority under this request."""

    bootstrap_archive_root(tmp_path)
    audit = AuditRepository.for_archive_root(tmp_path)
    principal = _principal()
    machine = MachineRequestBinding(
        "a" * 64, "request:foreign-plan", principal.actor_ref, "f" * 64, "maintenance.insights.rebuild"
    )
    with pytest.raises(ValueError, match=message):
        InsightAcceptance(audit, machine, principal).stage_preview(plan_factory())  # type: ignore[arg-type]
    assert audit.machine_request(machine) is None


def test_sealed_authority_can_start_only_through_its_reserved_machine_part(tmp_path: Path) -> None:
    """Removing the reservation check would let the generic staged bearer start first."""

    bootstrap_archive_root(tmp_path)
    audit = AuditRepository.for_archive_root(tmp_path)
    principal = _principal()
    targets = (AcceptedInsightTarget("session:one", "required"),)
    provisional = AcceptedInsightPart(
        "pending:one",
        "pending-auth:one",
        "0" * 64,
        0,
        1,
        "0" * 64,
        None,
        "explicit",
        "index-generation:/archive/fixture/index.db",
        "2",
        targets,
    )
    digest = insight_manifest_digest((provisional,))
    plan = _page(scope_kind="explicit", ordinal=0, count=1, digest=digest, previous_preview_ref=None, targets=targets)
    binding = runtime_operation_binding(InsightsRebuildActuator())
    machine = MachineRequestBinding(
        "a" * 64, "request:sealed-start", principal.actor_ref, "f" * 64, "maintenance.insights.rebuild"
    )
    acceptance = InsightAcceptance(audit, machine, principal)
    preview = acceptance.stage_preview(plan)
    authorization = acceptance.ensure_staged_authorization(OperationExecutor(audit=audit), binding, preview)
    accepted = acceptance.seal(
        head_preview_ref=preview.preview_ref,
        page_count=1,
        manifest_digest=digest,
        deadline_unix_ms=9_999_999_999_999,
    )

    with audit.bind_machine_request(machine, transition="consume_authorization_and_start", part=accepted[0].ordinal):
        operation_id = audit.consume_authorization_and_start(preview, authorization)
    assert operation_id.startswith("operation:")
    audit.finalize_attempt(
        operation_id,
        status="unknown",
        error_summary="test stopped before domain owner",
        unknown_reason="test only proves sealed start authority",
    )


def test_staged_preview_replay_retains_its_frozen_predecessor_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Minting a replay-only preview id would change the page-chain head and make this red."""

    bootstrap_archive_root(tmp_path)
    audit = AuditRepository.for_archive_root(tmp_path)
    principal = _principal()
    machine = MachineRequestBinding(
        "a" * 64, "request:replay-preview", principal.actor_ref, "a" * 64, "maintenance.insights.rebuild"
    )
    plan = _page(ordinal=0, count=1, digest="a" * 64, previous_preview_ref=None, targets=())
    original_phase = AuditContinuityCoordinator._phase

    def crash(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "append_insight_preview" and phase == "after_source_prepare":
            raise RuntimeError("synthetic append crash")
        original_phase(self, phase, mutation)

    with monkeypatch.context() as patch:
        patch.setattr(AuditContinuityCoordinator, "_phase", crash)
        with pytest.raises(RuntimeError, match="synthetic append crash"):
            with audit.bind_machine_request(machine, transition="append_insight_preview"):
                audit.append_insight_preview(plan, principal)
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        pending = json.loads(source.execute("SELECT pending_payload_json FROM audit_continuity_control").fetchone()[0])
    preview_ref = pending["command"]["payload"]["preview_id"]

    recovered = AuditRepository.for_archive_root(tmp_path)
    recovered.reconcile_continuity()

    assert recovered.machine_parts(machine)[0]["preview_ref"] == preview_ref
    assert recovered.preview_for_principal(preview_ref, principal).plan.plan_hash == plan.plan_hash

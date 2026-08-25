from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from polylogue.operations.bindings import (
    BindingValidationError,
    OperationBinding,
    OperationBindingCatalog,
    validate_operation_bindings,
)
from polylogue.operations.mutation_transaction import (
    ConfirmationStrength,
    DestructiveClass,
    MutationPlan,
    MutationReceipt,
    RecoveryDeclaration,
    RecoveryDisposition,
    TargetAuthorityPolicy,
    build_plan,
)
from polylogue.operations.specs import OperationKind, OperationSpec, build_runtime_operation_catalog


@dataclass
class _Actuator:
    operation: str = "mutate-fixture"
    operation_version: int = 3
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, _args: object) -> MutationPlan:
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=("session:fixture",),
            affected_tiers=("user",),
            reversible=True,
        )

    def apply(self, plan: MutationPlan, _args: object) -> MutationReceipt:
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

    def inspect_recovery(self, _operation: object, _args: object) -> RecoveryDisposition:
        return RecoveryDisposition("unknown", "operator-blocking", "synthetic inspector")


def _spec(*, policies: tuple[TargetAuthorityPolicy, ...] | None = None) -> OperationSpec:
    return OperationSpec(
        name="mutate-fixture",
        kind=OperationKind.MAINTENANCE,
        description="fixture",
        mutates_state=True,
        executor_status="executor-routed",
        operation_version=3,
        allowed_surfaces=("internal",),
        target_authority=(
            policies
            if policies is not None
            else (
                TargetAuthorityPolicy(
                    key="session",
                    target_kinds=("session",),
                    required_capabilities=("archive.fixture.write",),
                    destructive_class="reversible",
                    required_confirmation="role_only",
                    allowed_durabilities=("derived",),
                    allowed_recovery=("none",),
                ),
            )
        ),
        recovery=RecoveryDeclaration(
            target_identity="typed-targets-v1",
            plan_binding="plan-hash-and-target-digest-v1",
            precondition_inspection="domain-owned",
            postcondition_inspection="domain-owned",
            exact_retry=True,
            partial_actions=("retry-exact",),
            capability="inspect-and-retry",
        ),
    )


def test_binding_rejects_unregistered_capability_and_missing_policy() -> None:
    actuator = _Actuator()
    with pytest.raises(BindingValidationError, match="outside the archive capability namespace"):
        OperationBinding(
            _spec(
                policies=(
                    TargetAuthorityPolicy(
                        key="session",
                        target_kinds=("session",),
                        required_capabilities=("fake.capability",),
                        destructive_class="reversible",
                        required_confirmation="role_only",
                        allowed_durabilities=("derived",),
                        allowed_recovery=("none",),
                    ),
                )
            ),
            actuator,
        ).validate()
    with pytest.raises(BindingValidationError, match="no target authority"):
        OperationBinding(_spec(policies=()), actuator).validate()
    with pytest.raises(BindingValidationError, match="no recovery declaration"):
        OperationBinding(replace(_spec(), recovery=None), actuator).validate()


def test_binding_rejects_version_mismatch_and_duplicate_catalog_entries() -> None:
    binding: OperationBinding[object, object] = OperationBinding(_spec(), _Actuator())
    with pytest.raises(BindingValidationError, match="binding version"):
        OperationBinding(_spec(), _Actuator(), operation_version=2).validate()
    with pytest.raises(BindingValidationError, match="duplicate"):
        OperationBindingCatalog.validate((binding, binding))


def test_catalog_requires_every_executor_routed_spec_and_resolves_only_registered_actuators() -> None:
    binding: OperationBinding[object, object] = OperationBinding(_spec(), _Actuator())
    catalog = validate_operation_bindings((_spec(),), (binding,))
    assert catalog.resolve("mutate-fixture", 3) is binding
    with pytest.raises(BindingValidationError, match="no registered actuator"):
        catalog.resolve("mutate-other", 1)


def test_catalog_rejects_missing_binding() -> None:
    with pytest.raises(BindingValidationError, match="missing"):
        validate_operation_bindings((_spec(),), ())


def test_runtime_executor_routes_have_specific_capabilities_and_surfaces() -> None:
    specs = build_runtime_operation_catalog().by_name().values()
    routed = [spec for spec in specs if spec.executor_status == "executor-routed"]

    assert routed
    assert all(spec.target_authority for spec in routed)
    assert all(spec.recovery is not None for spec in routed)
    assert all(spec.allowed_surfaces for spec in routed)
    assert all(
        capability != "archive.legacy_runtime"
        for spec in routed
        for policy in spec.target_authority
        for capability in policy.required_capabilities
    )

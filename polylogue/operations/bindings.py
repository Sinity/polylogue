"""Validated OperationSpec to actuator bindings for mutation authority."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from polylogue.operations.mutation_transaction import MutationActuator
from polylogue.operations.specs import OperationSpec

ArgsT = TypeVar("ArgsT", contravariant=True)
ResultT = TypeVar("ResultT", covariant=True)


class BindingValidationError(ValueError):
    """A binding catalog cannot be made authoritative."""


@dataclass(frozen=True, slots=True)
class OperationBinding(Generic[ArgsT, ResultT]):
    """One versioned OperationSpec joined to exactly one actuator."""

    spec: OperationSpec
    actuator: MutationActuator[ArgsT]
    operation_version: int | None = None
    declared_capabilities: tuple[str, ...] | None = None

    def validate(self) -> None:
        if not self.spec.mutates_state:
            raise BindingValidationError(f"read-only operation cannot bind an actuator: {self.spec.name!r}")
        if self.spec.executor_status != "executor-routed":
            raise BindingValidationError(
                f"operation {self.spec.name!r} is not declared executor-routed: {self.spec.executor_status!r}"
            )
        if not self.spec.target_authority:
            raise BindingValidationError(f"operation {self.spec.name!r} has no target authority policy rows")
        actuator_operation = getattr(self.actuator, "operation", None)
        if actuator_operation != self.spec.name:
            raise BindingValidationError(
                f"actuator operation {actuator_operation!r} does not match spec {self.spec.name!r}"
            )
        actuator_version = getattr(self.actuator, "operation_version", self.spec.operation_version)
        if self.operation_version is not None and self.operation_version != self.spec.operation_version:
            raise BindingValidationError(
                f"binding version {self.operation_version} does not match spec version {self.spec.operation_version}"
            )
        if actuator_version != self.spec.operation_version:
            raise BindingValidationError(
                f"actuator version {actuator_version} does not match spec version {self.spec.operation_version}"
            )
        policy_capabilities = tuple(
            sorted({capability for policy in self.spec.target_authority for capability in policy.required_capabilities})
        )
        if self.declared_capabilities is not None and tuple(sorted(self.declared_capabilities)) != policy_capabilities:
            raise BindingValidationError("binding capabilities do not equal the spec-owned capability rows")
        if any(not capability.startswith("archive.") for capability in policy_capabilities):
            raise BindingValidationError("binding contains a capability outside the archive capability namespace")
        policy_keys = [policy.key for policy in self.spec.target_authority]
        if len(policy_keys) != len(set(policy_keys)):
            raise BindingValidationError(f"duplicate target authority policy for {self.spec.name!r}")


@dataclass(frozen=True, slots=True)
class OperationBindingCatalog:
    """Immutable validated binding inventory used by composition roots."""

    bindings: tuple[OperationBinding[object, object], ...]

    @classmethod
    def validate(cls, bindings: Iterable[OperationBinding[object, object]]) -> OperationBindingCatalog:
        materialized = tuple(bindings)
        names = [(binding.spec.name, binding.spec.operation_version) for binding in materialized]
        if len(names) != len(set(names)):
            raise BindingValidationError("duplicate OperationBinding for operation/version")
        for binding in materialized:
            binding.validate()
        return cls(materialized)

    def resolve(self, operation: str, operation_version: int = 1) -> OperationBinding[object, object]:
        for binding in self.bindings:
            if binding.spec.name == operation and binding.spec.operation_version == operation_version:
                return binding
        raise BindingValidationError(f"no registered actuator binding for {operation!r} v{operation_version}")


def validate_operation_bindings(
    specs: Iterable[OperationSpec], bindings: Iterable[OperationBinding[object, object]]
) -> OperationBindingCatalog:
    """Require exactly one binding for every executor-routed mutating spec."""

    catalog = OperationBindingCatalog.validate(bindings)
    required = {
        (spec.name, spec.operation_version)
        for spec in specs
        if spec.mutates_state and spec.executor_status == "executor-routed"
    }
    actual = {(binding.spec.name, binding.spec.operation_version) for binding in catalog.bindings}
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing or unexpected:
        raise BindingValidationError(f"binding inventory mismatch: missing={missing}, unexpected={unexpected}")
    return catalog


__all__ = [
    "BindingValidationError",
    "OperationBinding",
    "OperationBindingCatalog",
    "validate_operation_bindings",
]

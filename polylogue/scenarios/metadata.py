"""Shared scenario metadata used across compiled verification artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polylogue.artifacts import ArtifactNode
    from polylogue.operations import OperationSpec


def _coerce_string(value: object, default: str) -> str:
    if isinstance(value, str) and value:
        return value
    return default


def _coerce_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return ()


@lru_cache(maxsize=1)
def runtime_target_catalog() -> RuntimeTargetCatalog:
    from polylogue.artifacts import build_runtime_artifact_nodes
    from polylogue.operations import build_runtime_operation_specs

    return RuntimeTargetCatalog(
        artifacts=build_runtime_artifact_nodes(),
        operations=build_runtime_operation_specs(),
    )


@lru_cache(maxsize=1)
def runtime_artifact_target_names() -> tuple[str, ...]:
    return runtime_target_catalog().artifact_names()


@lru_cache(maxsize=1)
def runtime_operation_target_names() -> tuple[str, ...]:
    return runtime_target_catalog().operation_names()


@dataclass(frozen=True, slots=True)
class RuntimeTargetCatalog:
    """Resolved runtime artifact and operation targets shared across projections."""

    artifacts: tuple[ArtifactNode, ...]
    operations: tuple[OperationSpec, ...]

    def artifact_names(self) -> tuple[str, ...]:
        return tuple(artifact.name for artifact in self.artifacts)

    def operation_names(self) -> tuple[str, ...]:
        return tuple(operation.name for operation in self.operations)

    def resolve_artifacts(self, names: tuple[str, ...]) -> tuple[ArtifactNode, ...]:
        by_name = {artifact.name: artifact for artifact in self.artifacts}
        return tuple(by_name[name] for name in names if name in by_name)

    def resolve_operations(self, names: tuple[str, ...]) -> tuple[OperationSpec, ...]:
        by_name = {operation.name: operation for operation in self.operations}
        return tuple(by_name[name] for name in names if name in by_name)


@dataclass(frozen=True, kw_only=True)
class ScenarioMetadata:
    """Portable scenario metadata shared by exercises, campaigns, and reports."""

    origin: str = "authored"
    artifact_targets: tuple[str, ...] = ()
    operation_targets: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ScenarioMetadata:
        return cls(
            origin=_coerce_string(payload.get("origin"), "authored"),
            artifact_targets=_coerce_string_tuple(payload.get("artifact_targets")),
            operation_targets=_coerce_string_tuple(payload.get("operation_targets")),
            tags=_coerce_string_tuple(payload.get("tags")),
        )

    @classmethod
    def from_object(cls, obj: object) -> ScenarioMetadata:
        return cls(
            origin=_coerce_string(getattr(obj, "origin", None), "authored"),
            artifact_targets=_coerce_string_tuple(getattr(obj, "artifact_targets", None)),
            operation_targets=_coerce_string_tuple(getattr(obj, "operation_targets", None)),
            tags=_coerce_string_tuple(getattr(obj, "tags", None)),
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"origin": self.origin}
        if self.artifact_targets:
            payload["artifact_targets"] = list(self.artifact_targets)
        if self.operation_targets:
            payload["operation_targets"] = list(self.operation_targets)
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload

    def runtime_artifact_targets(self) -> tuple[str, ...]:
        return tuple(artifact.name for artifact in self.resolve_runtime_artifacts())

    def runtime_operation_targets(self) -> tuple[str, ...]:
        return tuple(operation.name for operation in self.resolve_runtime_operations())

    def resolve_runtime_artifacts(self) -> tuple[ArtifactNode, ...]:
        return runtime_target_catalog().resolve_artifacts(self.artifact_targets)

    def resolve_runtime_operations(self) -> tuple[OperationSpec, ...]:
        return runtime_target_catalog().resolve_operations(self.operation_targets)


__all__ = [
    "ScenarioMetadata",
    "RuntimeTargetCatalog",
    "runtime_artifact_target_names",
    "runtime_operation_target_names",
    "runtime_target_catalog",
]

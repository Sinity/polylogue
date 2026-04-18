"""Shared scenario metadata used across compiled verification artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from .payloads import (
    PayloadDict,
    merge_unique_string_tuples,
    payload_string,
    payload_string_tuple,
)

if TYPE_CHECKING:
    from polylogue.artifact_graph import ArtifactGraph
    from polylogue.artifacts import ArtifactNode, ArtifactPath
    from polylogue.maintenance_targets import MaintenanceTargetSpec
    from polylogue.operations import OperationSpec


def _partition_runtime_operation_targets(operation_targets: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    runtime_names = set(runtime_operation_target_names())
    runtime_targets: list[str] = []
    declared_only_targets: list[str] = []
    for target in operation_targets:
        if target in runtime_names:
            runtime_targets.append(target)
        else:
            declared_only_targets.append(target)
    return tuple(runtime_targets), tuple(declared_only_targets)


def _contains_non_runtime_artifact_targets(artifact_targets: tuple[str, ...]) -> bool:
    if not artifact_targets:
        return False
    runtime_names = set(runtime_artifact_target_names())
    return any(target not in runtime_names for target in artifact_targets)


def _object_string_attribute(obj: object, name: str, default: str) -> str:
    value = getattr(obj, name, None)
    if isinstance(value, str):
        return value
    return default


def _object_string_tuple_attribute(obj: object, name: str) -> tuple[str, ...]:
    value = getattr(obj, name, None)
    return payload_string_tuple(value) if isinstance(value, list | tuple) else ()


@lru_cache(maxsize=1)
def runtime_artifact_graph() -> ArtifactGraph:
    from polylogue.artifact_graph import build_artifact_graph

    return build_artifact_graph()


@lru_cache(maxsize=1)
def runtime_artifact_target_names() -> tuple[str, ...]:
    return runtime_artifact_graph().artifact_names()


@lru_cache(maxsize=1)
def runtime_operation_target_names() -> tuple[str, ...]:
    return runtime_artifact_graph().operation_names()


@lru_cache(maxsize=1)
def declared_operation_target_names() -> tuple[str, ...]:
    from polylogue.operations import build_declared_operation_catalog

    return build_declared_operation_catalog().names()


@lru_cache(maxsize=1)
def runtime_path_target_names() -> tuple[str, ...]:
    return runtime_artifact_graph().path_names()


@lru_cache(maxsize=1)
def runtime_maintenance_target_names() -> tuple[str, ...]:
    return runtime_artifact_graph().maintenance_target_names()


@dataclass(frozen=True, kw_only=True)
class ScenarioMetadata:
    """Portable scenario metadata shared by exercises, campaigns, and reports."""

    origin: str = "authored"
    path_targets: tuple[str, ...] = ()
    artifact_targets: tuple[str, ...] = ()
    operation_targets: tuple[str, ...] = ()
    maintenance_targets: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ScenarioMetadata:
        return cls(
            origin=payload_string(payload.get("origin"), "authored"),
            path_targets=payload_string_tuple(payload.get("path_targets")),
            artifact_targets=payload_string_tuple(payload.get("artifact_targets")),
            operation_targets=payload_string_tuple(payload.get("operation_targets")),
            maintenance_targets=payload_string_tuple(payload.get("maintenance_targets")),
            tags=payload_string_tuple(payload.get("tags")),
        )

    @classmethod
    def from_object(cls, obj: object) -> ScenarioMetadata:
        return cls(
            origin=_object_string_attribute(obj, "origin", "authored"),
            path_targets=_object_string_tuple_attribute(obj, "path_targets"),
            artifact_targets=_object_string_tuple_attribute(obj, "artifact_targets"),
            operation_targets=_object_string_tuple_attribute(obj, "operation_targets"),
            maintenance_targets=_object_string_tuple_attribute(obj, "maintenance_targets"),
            tags=_object_string_tuple_attribute(obj, "tags"),
        )

    def to_payload(self) -> PayloadDict:
        payload: PayloadDict = {"origin": self.origin}
        if self.path_targets:
            payload["path_targets"] = list(self.path_targets)
        if self.artifact_targets:
            payload["artifact_targets"] = list(self.artifact_targets)
        if self.operation_targets:
            payload["operation_targets"] = list(self.operation_targets)
        if self.maintenance_targets:
            payload["maintenance_targets"] = list(self.maintenance_targets)
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload

    def merged(self, *others: ScenarioMetadata) -> ScenarioMetadata:
        if not others:
            return self
        return ScenarioMetadata(
            origin=self.origin,
            path_targets=merge_unique_string_tuples(self.path_targets, *(other.path_targets for other in others)),
            artifact_targets=merge_unique_string_tuples(
                self.artifact_targets, *(other.artifact_targets for other in others)
            ),
            operation_targets=merge_unique_string_tuples(
                self.operation_targets, *(other.operation_targets for other in others)
            ),
            maintenance_targets=merge_unique_string_tuples(
                self.maintenance_targets, *(other.maintenance_targets for other in others)
            ),
            tags=merge_unique_string_tuples(self.tags, *(other.tags for other in others)),
        )

    def with_default_targets(self, defaults: ScenarioMetadata) -> ScenarioMetadata:
        explicit_runtime_operations, explicit_declared_only_operations = _partition_runtime_operation_targets(
            self.operation_targets
        )
        preserve_explicit_operations = bool(explicit_runtime_operations) or _contains_non_runtime_artifact_targets(
            self.artifact_targets
        )
        return ScenarioMetadata(
            origin=self.origin,
            path_targets=self.path_targets or defaults.path_targets,
            artifact_targets=self.artifact_targets or defaults.artifact_targets,
            operation_targets=(
                self.operation_targets
                if preserve_explicit_operations
                else merge_unique_string_tuples(
                    explicit_declared_only_operations,
                    explicit_runtime_operations or defaults.operation_targets,
                )
            ),
            maintenance_targets=self.maintenance_targets or defaults.maintenance_targets,
            tags=merge_unique_string_tuples(self.tags, defaults.tags),
        )

    def runtime_path_targets(self) -> tuple[str, ...]:
        return tuple(path.name for path in self.resolve_runtime_paths())

    def runtime_artifact_targets(self) -> tuple[str, ...]:
        return tuple(artifact.name for artifact in self.resolve_runtime_artifacts())

    def runtime_operation_targets(self) -> tuple[str, ...]:
        return tuple(operation.name for operation in self.resolve_runtime_operations())

    def runtime_maintenance_targets(self) -> tuple[str, ...]:
        return tuple(target.name for target in self.resolve_runtime_maintenance_targets())

    def declared_operation_targets(self) -> tuple[str, ...]:
        return tuple(operation.name for operation in self.resolve_declared_operations())

    def resolve_runtime_paths(self) -> tuple[ArtifactPath, ...]:
        return runtime_artifact_graph().resolve_paths(self.path_targets)

    def resolve_runtime_artifacts(self) -> tuple[ArtifactNode, ...]:
        return runtime_artifact_graph().resolve_artifacts(self.artifact_targets)

    def resolve_runtime_operations(self) -> tuple[OperationSpec, ...]:
        return runtime_artifact_graph().resolve_operations(self.operation_targets)

    def resolve_runtime_maintenance_targets(self) -> tuple[MaintenanceTargetSpec, ...]:
        return runtime_artifact_graph().resolve_maintenance_targets(self.maintenance_targets)

    def resolve_declared_operations(self) -> tuple[OperationSpec, ...]:
        from polylogue.operations import build_declared_operation_catalog

        return build_declared_operation_catalog().resolve(self.operation_targets)


__all__ = [
    "declared_operation_target_names",
    "ScenarioMetadata",
    "runtime_artifact_graph",
    "runtime_artifact_target_names",
    "runtime_maintenance_target_names",
    "runtime_operation_target_names",
    "runtime_path_target_names",
]

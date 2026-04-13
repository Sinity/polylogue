"""Shared scenario metadata used across compiled verification artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polylogue.artifact_graph import ArtifactGraph
    from polylogue.artifacts import ArtifactNode, ArtifactPath
    from polylogue.operations import OperationSpec


def _coerce_string(value: object, default: str) -> str:
    if isinstance(value, str) and value:
        return value
    return default


def _coerce_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return ()


def _merge_unique_string_tuples(*groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return tuple(merged)


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


@dataclass(frozen=True, kw_only=True)
class ScenarioMetadata:
    """Portable scenario metadata shared by exercises, campaigns, and reports."""

    origin: str = "authored"
    path_targets: tuple[str, ...] = ()
    artifact_targets: tuple[str, ...] = ()
    operation_targets: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ScenarioMetadata:
        return cls(
            origin=_coerce_string(payload.get("origin"), "authored"),
            path_targets=_coerce_string_tuple(payload.get("path_targets")),
            artifact_targets=_coerce_string_tuple(payload.get("artifact_targets")),
            operation_targets=_coerce_string_tuple(payload.get("operation_targets")),
            tags=_coerce_string_tuple(payload.get("tags")),
        )

    @classmethod
    def from_object(cls, obj: object) -> ScenarioMetadata:
        return cls(
            origin=_coerce_string(getattr(obj, "origin", None), "authored"),
            path_targets=_coerce_string_tuple(getattr(obj, "path_targets", None)),
            artifact_targets=_coerce_string_tuple(getattr(obj, "artifact_targets", None)),
            operation_targets=_coerce_string_tuple(getattr(obj, "operation_targets", None)),
            tags=_coerce_string_tuple(getattr(obj, "tags", None)),
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"origin": self.origin}
        if self.path_targets:
            payload["path_targets"] = list(self.path_targets)
        if self.artifact_targets:
            payload["artifact_targets"] = list(self.artifact_targets)
        if self.operation_targets:
            payload["operation_targets"] = list(self.operation_targets)
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload

    def merged(self, *others: ScenarioMetadata) -> ScenarioMetadata:
        if not others:
            return self
        return ScenarioMetadata(
            origin=self.origin,
            path_targets=_merge_unique_string_tuples(self.path_targets, *(other.path_targets for other in others)),
            artifact_targets=_merge_unique_string_tuples(
                self.artifact_targets, *(other.artifact_targets for other in others)
            ),
            operation_targets=_merge_unique_string_tuples(
                self.operation_targets, *(other.operation_targets for other in others)
            ),
            tags=_merge_unique_string_tuples(self.tags, *(other.tags for other in others)),
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
                else _merge_unique_string_tuples(
                    explicit_declared_only_operations,
                    explicit_runtime_operations or defaults.operation_targets,
                )
            ),
            tags=_merge_unique_string_tuples(self.tags, defaults.tags),
        )

    def runtime_path_targets(self) -> tuple[str, ...]:
        return tuple(path.name for path in self.resolve_runtime_paths())

    def runtime_artifact_targets(self) -> tuple[str, ...]:
        return tuple(artifact.name for artifact in self.resolve_runtime_artifacts())

    def runtime_operation_targets(self) -> tuple[str, ...]:
        return tuple(operation.name for operation in self.resolve_runtime_operations())

    def declared_operation_targets(self) -> tuple[str, ...]:
        return tuple(operation.name for operation in self.resolve_declared_operations())

    def resolve_runtime_paths(self) -> tuple[ArtifactPath, ...]:
        return runtime_artifact_graph().resolve_paths(self.path_targets)

    def resolve_runtime_artifacts(self) -> tuple[ArtifactNode, ...]:
        return runtime_artifact_graph().resolve_artifacts(self.artifact_targets)

    def resolve_runtime_operations(self) -> tuple[OperationSpec, ...]:
        return runtime_artifact_graph().resolve_operations(self.operation_targets)

    def resolve_declared_operations(self) -> tuple[OperationSpec, ...]:
        from polylogue.operations import build_declared_operation_catalog

        return build_declared_operation_catalog().resolve(self.operation_targets)


__all__ = [
    "declared_operation_target_names",
    "ScenarioMetadata",
    "runtime_artifact_graph",
    "runtime_artifact_target_names",
    "runtime_operation_target_names",
    "runtime_path_target_names",
]

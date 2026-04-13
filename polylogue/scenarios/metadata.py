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
    from polylogue.operations import build_declared_operation_specs

    return tuple(spec.name for spec in build_declared_operation_specs())


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
        from polylogue.operations import build_declared_operation_specs

        by_name = {operation.name: operation for operation in build_declared_operation_specs()}
        return tuple(by_name[name] for name in self.operation_targets if name in by_name)


__all__ = [
    "declared_operation_target_names",
    "ScenarioMetadata",
    "runtime_artifact_graph",
    "runtime_artifact_target_names",
    "runtime_operation_target_names",
    "runtime_path_target_names",
]

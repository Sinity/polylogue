"""Shared scenario metadata used across compiled verification artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _coerce_string(value: object, default: str) -> str:
    if isinstance(value, str) and value:
        return value
    return default


def _coerce_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return ()


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


__all__ = ["ScenarioMetadata"]

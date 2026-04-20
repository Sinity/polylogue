"""Typed models for synthetic corpus selection and batch generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from polylogue.schemas.synthetic.wire_formats import WireEncoding, WireFormat

if TYPE_CHECKING:
    from polylogue.schemas.synthetic.relations import RelationConstraintSolver
    from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator

SchemaScalar: TypeAlias = str | int | float | bool | None
SchemaValue: TypeAlias = SchemaScalar | list["SchemaValue"] | dict[str, "SchemaValue"]
SchemaRecord: TypeAlias = dict[str, SchemaValue]
SyntheticStyle: TypeAlias = Literal["default", "showcase"]


@dataclass(frozen=True)
class SyntheticSchemaSelection:
    provider: str
    package_version: str
    element_kind: str | None
    schema: SchemaRecord
    wire_format: WireFormat


@dataclass(frozen=True)
class SyntheticArtifact:
    raw_bytes: bytes
    message_count: int
    style: str


@dataclass(frozen=True)
class SyntheticGenerationReport:
    provider: str
    package_version: str
    element_kind: str | None
    wire_encoding: WireEncoding
    requested_count: int
    generated_count: int
    style: SyntheticStyle
    seed: int | None


@dataclass(frozen=True)
class SyntheticGenerationBatch:
    artifacts: list[SyntheticArtifact]
    report: SyntheticGenerationReport

    @property
    def raw_items(self) -> list[bytes]:
        return [artifact.raw_bytes for artifact in self.artifacts]


@dataclass(frozen=True)
class SyntheticWrittenBatch:
    batch: SyntheticGenerationBatch
    files: tuple[Path, ...]


@dataclass
class SyntheticGenerationState:
    relation_solver: RelationConstraintSolver
    semantic_generator: SemanticValueGenerator | None = None


__all__ = [
    "SchemaRecord",
    "SchemaScalar",
    "SchemaValue",
    "SyntheticArtifact",
    "SyntheticGenerationBatch",
    "SyntheticGenerationState",
    "SyntheticGenerationReport",
    "SyntheticSchemaSelection",
    "SyntheticStyle",
    "SyntheticWrittenBatch",
]

"""Typed models for synthetic corpus selection and batch generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.schemas.synthetic.wire_formats import WireFormat


@dataclass(frozen=True)
class SyntheticSchemaSelection:
    provider: str
    package_version: str
    element_kind: str | None
    schema: dict[str, Any]
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
    wire_encoding: str
    requested_count: int
    generated_count: int
    style: str
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


__all__ = [
    "SyntheticArtifact",
    "SyntheticGenerationBatch",
    "SyntheticGenerationReport",
    "SyntheticSchemaSelection",
    "SyntheticWrittenBatch",
]

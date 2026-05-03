"""Typed artifact descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from polylogue.core.json import JSONDocument, json_document


class ArtifactLayer(str, Enum):
    SOURCE = "source"
    DURABLE = "durable"
    DERIVED = "derived"
    INDEX = "index"
    PROJECTION = "projection"


@dataclass(frozen=True, slots=True)
class ArtifactNode:
    """One named artifact or projection in the Polylogue runtime graph."""

    name: str
    layer: ArtifactLayer
    description: str
    depends_on: tuple[str, ...] = ()
    code_refs: tuple[str, ...] = ()
    repair_targets: tuple[str, ...] = ()
    readiness_surfaces: tuple[str, ...] = ()

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "name": self.name,
                "layer": self.layer.value,
                "description": self.description,
                "depends_on": list(self.depends_on),
                "code_refs": list(self.code_refs),
                "repair_targets": list(self.repair_targets),
                "readiness_surfaces": list(self.readiness_surfaces),
            }
        )


@dataclass(frozen=True, slots=True)
class ArtifactPath:
    """One curated path through the artifact graph."""

    name: str
    description: str
    nodes: tuple[str, ...]

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "name": self.name,
                "description": self.description,
                "nodes": list(self.nodes),
            }
        )

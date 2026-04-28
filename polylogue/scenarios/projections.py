"""Shared scenario-projection metadata for control-plane surfaces."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum

from polylogue.products.authored_payloads import PayloadDict, PayloadMap

from .metadata import ScenarioMetadata


class ScenarioProjectionSourceKind(str, Enum):
    EXERCISE = "exercise"
    VALIDATION_FAMILY = "validation-family"
    VALIDATION_LANE = "validation-lane"
    MUTATION_CAMPAIGN = "mutation-campaign"
    BENCHMARK_CAMPAIGN = "benchmark-campaign"
    SYNTHETIC_BENCHMARK = "synthetic-benchmark"
    INFERRED_CORPUS = "inferred-corpus"
    INFERRED_CORPUS_SCENARIO = "inferred-corpus-scenario"


class ScenarioProjectionSource:
    """Protocol-style mixin for authored types that can project themselves."""

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        raise NotImplementedError

    @property
    def projection_name(self) -> str:
        raise NotImplementedError

    @property
    def projection_description(self) -> str:
        raise NotImplementedError

    def projection_source_payload(self) -> PayloadMap:
        return {}

    def to_projection_entry(self) -> ScenarioProjectionEntry:
        metadata = ScenarioMetadata.from_object(self)
        return ScenarioProjectionEntry(
            source_kind=self.projection_source_kind,
            name=self.projection_name,
            description=self.projection_description,
            origin=metadata.origin,
            path_targets=metadata.path_targets,
            artifact_targets=metadata.artifact_targets,
            operation_targets=metadata.operation_targets,
            tags=metadata.tags,
            source_payload=dict(self.projection_source_payload()),
        )


@dataclass(frozen=True)
class ScenarioProjectionEntry(ScenarioMetadata):
    source_kind: ScenarioProjectionSourceKind
    name: str
    description: str
    source_payload: PayloadDict = field(default_factory=dict)

    @classmethod
    def from_source(cls, source: ScenarioProjectionSource) -> ScenarioProjectionEntry:
        return source.to_projection_entry()

    def to_dict(self) -> PayloadDict:
        data = asdict(self)
        data["source_kind"] = self.source_kind.value
        return data


def compile_projection_entries(
    sources: Iterable[ScenarioProjectionSource],
) -> tuple[ScenarioProjectionEntry, ...]:
    return tuple(source.to_projection_entry() for source in sources)


__all__ = [
    "compile_projection_entries",
    "ScenarioProjectionEntry",
    "ScenarioProjectionSource",
    "ScenarioProjectionSourceKind",
]

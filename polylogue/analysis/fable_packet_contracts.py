"""Closed descriptive Fable packet evidence and report types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import ConfigDict

from polylogue.analysis.cohorts import CohortManifest

PacketStatus = Literal["complete", "not_supported"]


@dataclass(frozen=True)
class DelegationPacketRow:
    """Bounded structural evidence needed by the descriptive packet."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    delegation_ref: str
    evidence_basis: Literal["action", "edge"]
    mapping_state: str
    instruction_sha256: str | None


@dataclass(frozen=True)
class DelegationPacketLabel:
    """One accepted or candidate descriptive annotation with evidence spans."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    delegation_ref: str
    field: str
    value: str | None
    batch_id: str
    accepted: bool
    applicable: bool | None
    confidence: float | None
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class DescriptiveDistribution:
    """One accepted-label distribution with explicit denominator/missingness."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    field: str
    value: str
    count: int
    proportion: float
    denominator_n: int
    missing_n: int


@dataclass(frozen=True)
class FableDelegationPacket:
    """A private descriptive packet or a concrete fail-closed explanation."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    status: PacketStatus
    manifest_id: str
    population_count: int
    action_observed_count: int
    edge_only_count: int
    unresolved_count: int
    selected_refs: tuple[str, ...]
    annotation_schema_id: str | None
    annotation_batches: tuple[str, ...]
    distributions: tuple[DescriptiveDistribution, ...]
    disagreement_count: int
    adjudication_counts: tuple[tuple[str, int], ...]
    specimen_refs: tuple[str, ...]
    counterexample_refs: tuple[str, ...]
    limits: tuple[str, ...]
    not_supported_reasons: tuple[str, ...] = ()
    manifest: CohortManifest | None = None
    label_evidence_refs: tuple[tuple[str, tuple[str, ...]], ...] = ()
    aggregate_evidence_refs: tuple[str, ...] = ()

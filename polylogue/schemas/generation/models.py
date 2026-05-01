"""Typed stage models for schema generation and package assembly."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from polylogue.core.json import JSONDocument
from polylogue.schemas.observation import SchemaUnit
from polylogue.schemas.packages import SchemaPackageCatalog
from polylogue.schemas.registry import ClusterManifest

if TYPE_CHECKING:
    from polylogue.schemas.redaction_report import SchemaReport


@dataclass
class GenerationResult:
    """Result of schema generation."""

    provider: str
    schema: JSONDocument | None
    sample_count: int
    error: str | None = None
    redaction_report: SchemaReport | None = None
    versions: list[str] = field(default_factory=list)
    default_version: str | None = None
    cluster_count: int = 0
    package_count: int = 0
    artifact_counts: dict[str, int] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.schema is not None and self.error is None


@dataclass
class _ProviderBundle:
    result: GenerationResult
    catalog: SchemaPackageCatalog | None = None
    package_schemas: dict[str, dict[str, JSONDocument]] = field(default_factory=dict)
    manifest: ClusterManifest | None = None


@dataclass
class _ProfileSummary:
    artifact_kind: str
    profile_tokens: tuple[str, ...]
    dominant_keys: list[str]
    sample_count: int = 0
    schema_sample_count: int = 0
    representative_paths: list[str] = field(default_factory=list)


@dataclass
class _ClusterAccumulator:
    artifact_kind: str
    dominant_keys: list[str]
    sample_count: int = 0
    schema_sample_count: int = 0
    representative_paths: list[str] = field(default_factory=list)
    profile_token_counts: Counter[str] = field(default_factory=Counter)
    member_profiles: set[tuple[str, ...]] = field(default_factory=set)
    reservoir_samples: list[Mapping[str, object]] = field(default_factory=list)
    reservoir_conv_ids: list[str | None] = field(default_factory=list)
    rng: random.Random = field(default_factory=lambda: random.Random(42))
    exact_structure_ids: set[str] = field(default_factory=set)
    bundle_scopes: set[str] = field(default_factory=set)
    first_seen: str | None = None
    last_seen: str | None = None


@dataclass(frozen=True)
class _UnitMembership:
    unit: SchemaUnit
    profile_family_id: str


@dataclass
class _PackageAccumulator:
    provider: str
    anchor_family_id: str
    anchor_kind: str
    memberships: list[_UnitMembership] = field(default_factory=list)
    bundle_scopes: set[str] = field(default_factory=set)
    representative_paths: list[str] = field(default_factory=list)
    profile_family_ids: set[str] = field(default_factory=set)
    first_seen: str | None = None
    last_seen: str | None = None


@dataclass(frozen=True)
class ClusterCollectionResult:
    clusters: dict[str, _ClusterAccumulator]
    memberships: list[_UnitMembership]
    sample_count: int
    artifact_counts: dict[str, int]


@dataclass(frozen=True)
class PackageAssemblyResult:
    packages: list[_PackageAccumulator]
    orphan_adjunct_counts: dict[str, int]


__all__ = [
    "ClusterCollectionResult",
    "GenerationResult",
    "PackageAssemblyResult",
    "_ClusterAccumulator",
    "_PackageAccumulator",
    "_ProfileSummary",
    "_ProviderBundle",
    "_UnitMembership",
]

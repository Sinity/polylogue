"""Typed operator requests and results for schema/evidence workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polylogue.schemas.generation_models import GenerationResult
from polylogue.schemas.packages import SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.tooling_registry import ClusterManifest, SchemaDiff
from polylogue.schemas.verification_models import ArtifactProofReport
from polylogue.storage.store import ArtifactCohortSummary, ArtifactObservationRecord


@dataclass(frozen=True)
class SchemaInferRequest:
    provider: str
    db_path: Path
    max_samples: int | None = None
    privacy_config: Any | None = None
    cluster: bool = False
    cluster_sample_limit: int = 500


@dataclass(frozen=True)
class SchemaInferResult:
    generation: GenerationResult
    manifest: ClusterManifest | None = None
    manifest_path: Path | None = None


@dataclass(frozen=True)
class SchemaListRequest:
    provider: str | None = None


@dataclass(frozen=True)
class SchemaProviderSnapshot:
    provider: str
    versions: list[str]
    catalog: SchemaPackageCatalog | None = None
    manifest: ClusterManifest | None = None
    latest_age_days: int | None = None


@dataclass(frozen=True)
class SchemaListResult:
    provider: str | None
    selected: SchemaProviderSnapshot | None = None
    providers: list[SchemaProviderSnapshot] = field(default_factory=list)


@dataclass(frozen=True)
class SchemaCompareRequest:
    provider: str
    from_version: str
    to_version: str
    element_kind: str | None = None


@dataclass(frozen=True)
class SchemaCompareResult:
    diff: SchemaDiff


@dataclass(frozen=True)
class SchemaPromoteRequest:
    provider: str
    cluster_id: str
    db_path: Path
    with_samples: bool = False
    max_samples: int = 500


@dataclass(frozen=True)
class SchemaPromoteResult:
    provider: str
    cluster_id: str
    package_version: str
    package: SchemaVersionPackage | None
    schema: dict[str, Any] | None
    versions: list[str]


@dataclass(frozen=True)
class SchemaRoleAssignment:
    path: str
    role: str
    confidence: float
    evidence: dict[str, Any]


@dataclass(frozen=True)
class SchemaCoverageSummary:
    total_fields: int
    with_format: int
    with_values: int
    with_role: int


@dataclass(frozen=True)
class SchemaAnnotationSummary:
    semantic_count: int
    format_count: int
    values_count: int
    total_enum_values: int
    roles: list[SchemaRoleAssignment]
    coverage: SchemaCoverageSummary


@dataclass(frozen=True)
class SchemaExplainRequest:
    provider: str
    version: str = "latest"
    element_kind: str | None = None


@dataclass(frozen=True)
class SchemaExplainResult:
    provider: str
    version: str
    element_kind: str | None
    package: SchemaVersionPackage | None
    schema: dict[str, Any]
    annotations: SchemaAnnotationSummary


@dataclass(frozen=True)
class SchemaAuditRequest:
    provider: str | None = None


@dataclass(frozen=True)
class ArtifactObservationListResult:
    rows: list[ArtifactObservationRecord]


@dataclass(frozen=True)
class ArtifactCohortListResult:
    rows: list[ArtifactCohortSummary]


@dataclass(frozen=True)
class ArtifactProofResult:
    report: ArtifactProofReport


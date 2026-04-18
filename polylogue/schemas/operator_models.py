"""Typed operator requests and results for schema/evidence workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polylogue.scenarios import CorpusScenario, CorpusSpec
from polylogue.schemas.generation_models import GenerationResult
from polylogue.schemas.packages import SchemaPackageCatalog, SchemaResolution, SchemaVersionPackage
from polylogue.schemas.tooling_registry import ClusterManifest, SchemaDiff
from polylogue.schemas.verification_models import ArtifactProofReport
from polylogue.storage.state_views import ArtifactCohortSummary
from polylogue.storage.store import ArtifactObservationRecord


@dataclass(frozen=True)
class SchemaInferRequest:
    provider: str
    db_path: Path
    max_samples: int | None = None
    privacy_config: Any | None = None
    cluster: bool = False
    cluster_sample_limit: int = 500
    full_corpus: bool = False


@dataclass(frozen=True)
class SchemaInferResult:
    generation: GenerationResult
    manifest: ClusterManifest | None = None
    manifest_path: Path | None = None
    corpus_specs: tuple[CorpusSpec, ...] = ()
    corpus_scenarios: tuple[CorpusScenario, ...] = ()


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
    corpus_specs: tuple[CorpusSpec, ...] = ()
    corpus_scenarios: tuple[CorpusScenario, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"provider": self.provider, "versions": list(self.versions)}
        if self.catalog is not None:
            payload["catalog"] = self.catalog.to_dict()
        if self.manifest is not None:
            payload["manifest"] = self.manifest.to_dict()
        if self.corpus_specs:
            payload["corpus_specs"] = [spec.to_payload() for spec in self.corpus_specs]
        if self.corpus_scenarios:
            payload["corpus_scenarios"] = [
                {
                    "provider": scenario.provider,
                    "package_version": scenario.package_version,
                    "corpus_specs": [spec.to_payload() for spec in scenario.corpus_specs],
                }
                for scenario in self.corpus_scenarios
            ]
        return payload

    def to_list_item_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "versions": list(self.versions),
            "package_count": len(self.catalog.packages) if self.catalog is not None else 0,
            "default_version": self.catalog.default_version if self.catalog is not None else None,
            "latest_version": self.catalog.latest_version if self.catalog is not None else None,
            "cluster_count": len(self.manifest.clusters) if self.manifest is not None else 0,
            "corpus_spec_count": len(self.corpus_specs),
            "corpus_scenario_count": len(self.corpus_scenarios),
        }


@dataclass(frozen=True)
class SchemaListResult:
    provider: str | None
    selected: SchemaProviderSnapshot | None = None
    providers: list[SchemaProviderSnapshot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any] | list[dict[str, Any]]:
        if self.provider is not None:
            if self.selected is None:
                return {"provider": self.provider, "versions": []}
            return self.selected.to_dict()
        return [snapshot.to_list_item_dict() for snapshot in self.providers]


@dataclass(frozen=True)
class SchemaCompareRequest:
    provider: str
    from_version: str
    to_version: str
    element_kind: str | None = None


@dataclass(frozen=True)
class SchemaCompareResult:
    diff: SchemaDiff

    def to_dict(self) -> dict[str, Any]:
        return self.diff.to_dict()


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "cluster_id": self.cluster_id,
            "package_version": self.package_version,
            "package": self.package.to_dict() if self.package is not None else None,
            "schema": self.schema,
        }


@dataclass(frozen=True)
class SchemaRoleAssignment:
    path: str
    role: str
    confidence: float
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "score": self.confidence,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class SchemaCoverageSummary:
    total_fields: int
    with_format: int
    with_values: int
    with_role: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_fields": self.total_fields,
            "with_format": self.with_format,
            "with_values": self.with_values,
            "with_role": self.with_role,
        }


@dataclass(frozen=True)
class SchemaAnnotationSummary:
    semantic_count: int
    format_count: int
    values_count: int
    total_enum_values: int
    roles: list[SchemaRoleAssignment]
    coverage: SchemaCoverageSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_count": self.semantic_count,
            "format_count": self.format_count,
            "values_count": self.values_count,
            "total_enum_values": self.total_enum_values,
            "roles": [role.to_dict() for role in self.roles],
            "coverage": self.coverage.to_dict(),
        }


@dataclass(frozen=True)
class SchemaRoleProofEntry:
    """Proof surface for a single semantic role assignment decision."""

    role: str
    chosen_path: str | None
    chosen_score: float
    competing: list[dict[str, Any]]
    evidence: dict[str, Any]
    abstained: bool
    abstain_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "chosen_path": self.chosen_path,
            "chosen_score": self.chosen_score,
            "competing": self.competing,
            "evidence": self.evidence,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
        }


@dataclass(frozen=True)
class SchemaReviewProof:
    """Full proof surface for all semantic role assignment decisions."""

    roles: list[SchemaRoleProofEntry]
    artifact_kind: str | None
    eligible_roles: list[str]
    ineligible_roles: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "roles": [entry.to_dict() for entry in self.roles],
            "artifact_kind": self.artifact_kind,
            "eligible_roles": self.eligible_roles,
            "ineligible_roles": self.ineligible_roles,
        }


@dataclass(frozen=True)
class SchemaExplainRequest:
    provider: str
    version: str = "latest"
    element_kind: str | None = None
    proof: bool = False


@dataclass(frozen=True)
class SchemaExplainResult:
    provider: str
    version: str
    element_kind: str | None
    package: SchemaVersionPackage | None
    schema: dict[str, Any]
    annotations: SchemaAnnotationSummary
    review_proof: SchemaReviewProof | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"schema": self.schema, "annotations": self.annotations.to_dict()}
        if self.package is not None:
            payload["package"] = self.package.to_dict()
        if self.review_proof is not None:
            payload["review_proof"] = self.review_proof.to_dict()
        return payload


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


@dataclass(frozen=True)
class SchemaPayloadResolveRequest:
    provider: str
    payload: dict[str, Any]
    source_path: str | None = None


@dataclass(frozen=True)
class SchemaPayloadResolveResult:
    provider: str
    source_path: str | None
    resolution: SchemaResolution | None

    @property
    def is_resolved(self) -> bool:
        return self.resolution is not None

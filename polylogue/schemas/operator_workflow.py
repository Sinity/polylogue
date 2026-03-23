"""Typed operator workflows for schema and evidence commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polylogue.schemas.operator_models import (
    ArtifactCohortListResult,
    ArtifactObservationListResult,
    ArtifactProofResult,
    SchemaAnnotationSummary,
    SchemaAuditRequest,
    SchemaCompareRequest,
    SchemaCompareResult,
    SchemaCoverageSummary,
    SchemaExplainRequest,
    SchemaExplainResult,
    SchemaInferRequest,
    SchemaInferResult,
    SchemaListRequest,
    SchemaListResult,
    SchemaPromoteRequest,
    SchemaPromoteResult,
    SchemaProviderSnapshot,
    SchemaRoleAssignment,
)
from polylogue.schemas.verification_artifacts import (
    list_artifact_cohort_rows,
    list_artifact_observation_rows,
    prove_raw_artifact_coverage,
)
from polylogue.schemas.verification_corpus import verify_raw_corpus
from polylogue.schemas.verification_models import SchemaVerificationReport
from polylogue.schemas.verification_requests import (
    ArtifactObservationQuery,
    ArtifactProofRequest,
    SchemaVerificationRequest,
)


def infer_schema(request: SchemaInferRequest) -> SchemaInferResult:
    from polylogue.schemas.generation_workflow import generate_provider_schema
    from polylogue.schemas.observation import PROVIDERS
    from polylogue.schemas.registry import SchemaRegistry
    from polylogue.schemas.sampling import load_samples_from_db

    result = generate_provider_schema(
        request.provider,
        db_path=request.db_path,
        max_samples=request.max_samples,
        privacy_config=request.privacy_config,
    )
    if not request.cluster or not result.success:
        return SchemaInferResult(generation=result)

    config = PROVIDERS.get(request.provider)
    if config is None or not config.db_provider_name:
        return SchemaInferResult(generation=result)

    samples = load_samples_from_db(
        config.db_provider_name,
        db_path=request.db_path,
        max_samples=request.max_samples or request.cluster_sample_limit,
    )
    if not samples:
        return SchemaInferResult(generation=result)

    registry = SchemaRegistry()
    manifest = registry.cluster_samples(request.provider, samples)
    manifest_path = registry.save_cluster_manifest(manifest)
    return SchemaInferResult(
        generation=result,
        manifest=manifest,
        manifest_path=manifest_path,
    )


def list_schemas(request: SchemaListRequest) -> SchemaListResult:
    from polylogue.schemas.registry import SchemaRegistry

    registry = SchemaRegistry()
    if request.provider is not None:
        provider = request.provider
        return SchemaListResult(
            provider=provider,
            selected=SchemaProviderSnapshot(
                provider=provider,
                versions=registry.list_versions(provider),
                catalog=registry.load_package_catalog(provider),
                manifest=registry.load_cluster_manifest(provider),
                latest_age_days=registry.get_schema_age_days(provider),
            ),
        )

    snapshots = [
        SchemaProviderSnapshot(
            provider=provider,
            versions=registry.list_versions(provider),
            catalog=registry.load_package_catalog(provider),
            manifest=registry.load_cluster_manifest(provider),
            latest_age_days=registry.get_schema_age_days(provider),
        )
        for provider in registry.list_providers()
    ]
    return SchemaListResult(provider=None, providers=snapshots)


def compare_schema_versions(request: SchemaCompareRequest) -> SchemaCompareResult:
    from polylogue.schemas.registry import SchemaRegistry

    registry = SchemaRegistry()
    return SchemaCompareResult(
        diff=registry.compare_versions(
            request.provider,
            request.from_version,
            request.to_version,
            element_kind=request.element_kind,
        )
    )


def promote_schema_cluster(request: SchemaPromoteRequest) -> SchemaPromoteResult:
    from polylogue.schemas.observation import PROVIDERS, fingerprint_hash
    from polylogue.schemas.registry import SchemaRegistry
    from polylogue.schemas.sampling import load_samples_from_db
    from polylogue.schemas.shape_fingerprint import _structure_fingerprint

    registry = SchemaRegistry()
    samples: list[dict[str, Any]] | None = None
    if request.with_samples:
        config = PROVIDERS.get(request.provider)
        if config is None:
            raise ValueError(f"Unknown provider: {request.provider}")
        if config.db_provider_name:
            all_samples = load_samples_from_db(
                config.db_provider_name,
                db_path=request.db_path,
                max_samples=request.max_samples,
            )
        else:
            all_samples = []
        samples = [
            sample
            for sample in all_samples
            if fingerprint_hash(_structure_fingerprint(sample)) == request.cluster_id
        ]
        if not samples:
            raise ValueError(f"No samples match cluster {request.cluster_id}")

    new_version = registry.promote_cluster(
        request.provider,
        request.cluster_id,
        samples=samples,
    )
    return SchemaPromoteResult(
        provider=request.provider,
        cluster_id=request.cluster_id,
        package_version=new_version,
        package=registry.get_package(request.provider, version=new_version),
        schema=registry.get_element_schema(request.provider, version=new_version),
        versions=registry.list_versions(request.provider),
    )


def _collect_annotation_summary(schema: dict[str, Any]) -> SchemaAnnotationSummary:
    semantic_count = 0
    format_count = 0
    values_count = 0
    total_enum_values = 0
    roles: list[SchemaRoleAssignment] = []
    total_fields = 0
    with_format = 0
    with_values = 0
    with_role = 0

    def _visit(node: dict[str, Any], *, path: str) -> None:
        nonlocal semantic_count, format_count, values_count, total_enum_values
        nonlocal total_fields, with_format, with_values, with_role
        if not isinstance(node, dict):
            return
        role = node.get("x-polylogue-semantic-role")
        if role:
            semantic_count += 1
            roles.append(
                SchemaRoleAssignment(
                    path=path,
                    role=str(role),
                    confidence=float(node.get("x-polylogue-confidence", 0.0) or 0.0),
                    evidence=dict(node.get("x-polylogue-evidence", {})),
                )
            )
        if "x-polylogue-format" in node:
            format_count += 1
        if "x-polylogue-values" in node:
            values_count += 1
            total_enum_values += len(node["x-polylogue-values"])

        for name, child in node.get("properties", {}).items():
            if isinstance(child, dict):
                total_fields += 1
                if "x-polylogue-format" in child:
                    with_format += 1
                if "x-polylogue-values" in child:
                    with_values += 1
                if "x-polylogue-semantic-role" in child:
                    with_role += 1
                _visit(child, path=f"{path}.{name}")
        if isinstance(node.get("items"), dict):
            _visit(node["items"], path=f"{path}[*]")
        if isinstance(node.get("additionalProperties"), dict):
            _visit(node["additionalProperties"], path=f"{path}.*")
        for keyword in ("anyOf", "oneOf", "allOf"):
            for child in node.get(keyword, []):
                if isinstance(child, dict):
                    _visit(child, path=path)

    _visit(schema, path="$")
    return SchemaAnnotationSummary(
        semantic_count=semantic_count,
        format_count=format_count,
        values_count=values_count,
        total_enum_values=total_enum_values,
        roles=sorted(roles, key=lambda item: (-item.confidence, item.path, item.role)),
        coverage=SchemaCoverageSummary(
            total_fields=total_fields,
            with_format=with_format,
            with_values=with_values,
            with_role=with_role,
        ),
    )


def explain_schema(request: SchemaExplainRequest) -> SchemaExplainResult:
    from polylogue.schemas.registry import SchemaRegistry

    registry = SchemaRegistry()
    package = registry.get_package(request.provider, version=request.version)
    schema = registry.get_element_schema(
        request.provider,
        version=request.version,
        element_kind=request.element_kind,
    )
    if schema is None:
        raise ValueError(
            f"No schema found for {request.provider} version={request.version}"
            + (f" element={request.element_kind}" if request.element_kind else "")
        )
    return SchemaExplainResult(
        provider=request.provider,
        version=request.version,
        element_kind=request.element_kind,
        package=package,
        schema=schema,
        annotations=_collect_annotation_summary(schema),
    )


def audit_schemas(request: SchemaAuditRequest):
    from polylogue.schemas.audit import audit_all_providers, audit_provider

    return audit_provider(request.provider) if request.provider else audit_all_providers()


def run_schema_verification(request: SchemaVerificationRequest, *, db_path: Path) -> SchemaVerificationReport:
    return verify_raw_corpus(db_path=db_path, request=request)


def run_artifact_proof(request: ArtifactProofRequest, *, db_path: Path) -> ArtifactProofResult:
    return ArtifactProofResult(report=prove_raw_artifact_coverage(db_path=db_path, request=request))


def list_artifact_observations(
    request: ArtifactObservationQuery,
    *,
    db_path: Path,
) -> ArtifactObservationListResult:
    return ArtifactObservationListResult(
        rows=list_artifact_observation_rows(db_path=db_path, request=request),
    )


def list_artifact_cohorts(
    request: ArtifactObservationQuery,
    *,
    db_path: Path,
) -> ArtifactCohortListResult:
    return ArtifactCohortListResult(
        rows=list_artifact_cohort_rows(db_path=db_path, request=request),
    )


"""Inference/listing/promotion helpers for schema operator workflows."""

from __future__ import annotations

from typing import Any

from polylogue.schemas.operator_models import (
    SchemaCompareRequest,
    SchemaCompareResult,
    SchemaInferRequest,
    SchemaInferResult,
    SchemaListRequest,
    SchemaListResult,
    SchemaPromoteRequest,
    SchemaPromoteResult,
    SchemaProviderSnapshot,
)
from polylogue.schemas.operator_registry import schema_registry


def infer_schema(request: SchemaInferRequest) -> SchemaInferResult:
    from polylogue.schemas.generation_workflow import generate_provider_schema
    from polylogue.schemas.observation import PROVIDERS
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

    registry = schema_registry()
    manifest = registry.cluster_samples(request.provider, samples)
    manifest_path = registry.save_cluster_manifest(manifest)
    return SchemaInferResult(
        generation=result,
        manifest=manifest,
        manifest_path=manifest_path,
    )


def list_schemas(request: SchemaListRequest) -> SchemaListResult:
    registry = schema_registry()
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
    registry = schema_registry()
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
    from polylogue.schemas.sampling import load_samples_from_db
    from polylogue.schemas.shape_fingerprint import _structure_fingerprint

    registry = schema_registry()
    samples: list[dict[str, Any]] | None = None
    if request.with_samples:
        config = PROVIDERS.get(request.provider)
        if config is None:
            raise ValueError(f"Unknown provider: {request.provider}")
        all_samples = (
            load_samples_from_db(
                config.db_provider_name,
                db_path=request.db_path,
                max_samples=request.max_samples,
            )
            if config.db_provider_name
            else []
        )
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


def audit_schemas(request):
    from polylogue.schemas.audit_workflow import audit_all_providers, audit_provider

    return audit_provider(request.provider) if request.provider else audit_all_providers()


__all__ = [
    "audit_schemas",
    "compare_schema_versions",
    "infer_schema",
    "list_schemas",
    "promote_schema_cluster",
]
